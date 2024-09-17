import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

import dataclasses
import json
from typing import Dict, List, Optional, Union
import numpy as np
import gcsfs
from google.cloud import storage
import io
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    is_torch_tpu_available
)

from ezcolorlog import root_logger as logger
from cambrian.utils import IS_XLA_AVAILABLE

from packaging import version
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
from typing import List, Optional

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import is_apex_available
if is_apex_available():
    from apex import amp

import random
fs = gcsfs.GCSFileSystem(project='nyu-vision-lab')

HOME_DIR = os.path.expanduser("~") + "/"
print("HOME_DIR = ", HOME_DIR)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


def _fetch_gradients(optimizer, param_to_name, selected_module_names):
    gradients = []
    for param_group in optimizer.param_groups:
        for group, params in param_group.items():
            if group == 'params':
                for p in params:
                    # Use the mapping to get the module name
                    module_name = param_to_name.get(p, "")
                    # Check if the module name matches your criteria
                    if isinstance(p, torch.Tensor) and p.grad is not None and any(selected_name in module_name for selected_name in selected_module_names):
                        p.grad = p.grad.to(torch.float32)
                        gradients.append(p.grad.data)
    return gradients

REDUCE_SUM = 'sum'

def reduce_gradients(optimizer, param_to_name, selected_module_names, groups=None):
    # Initialize the process group if not already done
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    
    world_size = dist.get_world_size()
    
    if world_size > 1:
        gradients = _fetch_gradients(optimizer, param_to_name, selected_module_names)
        
        for grad in gradients:
            dist.all_reduce(grad, op=dist.ReduceOp.SUM)
            grad /= world_size

def _fetch_gradients(optimizer, param_to_name, selected_module_names):
    gradients = []
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None and param_to_name[param] in selected_module_names:
                gradients.append(param.grad.data)
    return gradients

def map_params_to_module_names(model_list):
    param_to_name = {}
    for model in model_list:
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                param_to_name[param] = f"{module_name}.{param_name}"
    return param_to_name


class CambrianTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        selected_module_names = ['vision_tower']
        # if self.args.unfreeze_mm_vision_tower:
        #     reduce_gradients(self.optimizer, self.param_to_name, selected_module_names)
        return loss.detach() / self.args.gradient_accumulation_steps

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        opt_model = self.model
        # if self.args.unfreeze_mm_vision_tower:
        #     opt_model.get_model().vision_tower_aux_list = nn.ModuleList(opt_model.get_vision_tower_aux_list())
        #     self.param_to_name = map_params_to_module_names([opt_model])
        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            assert not (self.args.mm_projector_lr and self.args.mm_vision_sampler_lr)
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.mm_vision_sampler_lr is not None:
                vision_sampler_parameters = [name for name, _ in opt_model.named_parameters() if ("vision_sampler" in name) or ("vision_query" in name) ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_sampler_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_sampler_lr,
                    },
                ]
            elif self.args.unfreeze_mm_vision_tower and self.args.mm_vision_tower_lr is not None:
                vision_tower_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in vision_tower_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        return self.optimizer
    

    def remove_prefix(text, prefix='gs://us-central2-storage/'):
        if prefix in text:
            return text.replace(prefix, '')
        return text
    
    def _load_rng_state(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # remove local path prefix if exists
        HOME_DIR = os.path.expanduser("~")  # 假设 HOME_DIR 是用户主目录
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        # Initialize the distributed process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')

        # get worker details
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # get path
        RNG_NAME = f'rng_rank-{rank:08d}-of-{world_size:08d}-rng.pth'
        RNG_PATH = os.path.join(resume_from_checkpoint_clean, RNG_NAME)

        # Loading the model weights:
        client = storage.Client()
        bucket = client.get_bucket('us-central2-storage')
        blob = bucket.blob(RNG_PATH)
        blob_bytes = blob.download_as_bytes()
        buffer = io.BytesIO(blob_bytes)
        rng_dict = torch.load(buffer)

        # Setting the seeds correctly
        random.setstate(rng_dict["python"])
        np.random.set_state(rng_dict["numpy"])
        torch.random.set_rng_state(rng_dict["cpu"])
        torch.cuda.set_rng_state(rng_dict["gpu"])  # Assuming GPU state is saved under "gpu"
        print("rng state loaded")

        # Destroy the process group if no longer needed
        dist.destroy_process_group()

    def _load_optimizer_and_scheduler(self, resume_from_checkpoint):
        if resume_from_checkpoint is None:
            return

        # remove local path prefix
        HOME_DIR = os.path.expanduser("~")  # 假设 HOME_DIR 是用户主目录
        if HOME_DIR in resume_from_checkpoint:
            resume_from_checkpoint_clean = resume_from_checkpoint.replace(HOME_DIR, '')
        else:
            resume_from_checkpoint_clean = resume_from_checkpoint

        # Initialize the distributed process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')

        # get worker details
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # get path to file
        WEIGHTS_NAME = "pytorch_model.bin"
        SCHEDULER_NAME = "scheduler.pt"
        SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_PATH = os.path.join(resume_from_checkpoint_clean, SHARD_NAME_OPT)
        LR_PATH = os.path.join(resume_from_checkpoint_clean, SCHEDULER_NAME)

        # connect to gcloud bucket
        client = storage.Client()
        bucket = client.get_bucket('us-central2-storage')

        # Loading opt state to each device
        blob = bucket.blob(SHARD_NAME_PATH)
        blob_bytes = blob.download_as_bytes()
        buffer = io.BytesIO(blob_bytes)
        optimizer_state = torch.load(buffer, map_location="cpu")
        optimizer_state = optimizer_state['optimizer_state']

        # Loading the schedule to each device
        blob_lr = bucket.blob(LR_PATH)
        blob_bytes_lr = blob_lr.download_as_bytes()
        buffer_lr = io.BytesIO(blob_bytes_lr)
        lr_scheduler_state = torch.load(buffer_lr, map_location="cpu")

        # No need for this, since already inside XLA spawn?
        # xm.send_cpu_data_to_device(optimizer_state, self.args.device)
        # xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

        # Move states to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer_state = {k: v.to(device) for k, v in optimizer_state.items()}
        lr_scheduler_state = {k: v.to(device) for k, v in lr_scheduler_state.items()}

        # Load state
        self.optimizer.load_state_dict(optimizer_state)
        self.lr_scheduler.load_state_dict(lr_scheduler_state)

        logger.info(f"Optimizer state and scheduler successfully loaded from {SHARD_NAME_PATH}")
        print("Loaded optimizer state successfully")

    def _save_checkpoint(self, model, trial, metrics=None):
        # Initialize the distributed process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')

        # get worker details
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Names of files
        TRAINING_ARGS_NAME = "training_args.bin"
        WEIGHTS_NAME = "pytorch_model.bin"
        SCHEDULER_NAME = "scheduler.pt"
        TRAINER_STATE_NAME = "trainer_state.json"

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model = self.model

        # Name of files to save
        SHARD_NAME = f'weights_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        SHARD_NAME_OPT = f'opt_rank-{rank:08d}-of-{world_size:08d}-{WEIGHTS_NAME}'
        RNG_NAME = f'rng_rank-{rank:08d}-of-{world_size:08d}-rng.pth'

        # Path of files to save
        SHARD_NAME_PATH = os.path.join(output_dir, SHARD_NAME)
        SHARD_NAME_OPT_PATH = os.path.join(output_dir, SHARD_NAME_OPT)
        LR_PATH = os.path.join(output_dir, SCHEDULER_NAME)
        TRAIN_ARGS_PATH = os.path.join(output_dir, TRAINING_ARGS_NAME)
        TRAINER_STATE_NAME_PATH = os.path.join(output_dir, TRAINER_STATE_NAME)
        RNG_PATH = os.path.join(output_dir, RNG_NAME)
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()

        # Final form of model and opt
        ckpt = {
            'model': self.model.state_dict(),
            'shard_metadata': self.model.get_shard_metadata()
        }
        opt_ckpt = {
            'optimizer_state' : self.optimizer.state_dict(),
            'shard_metadata': self.model.get_shard_metadata()
        }

        # Saving model shards
        with open(SHARD_NAME_PATH, 'wb') as f:
            torch.save(ckpt, f)

        # Saving optimizer shards
        with open(SHARD_NAME_OPT_PATH, 'wb') as f:
            torch.save(opt_ckpt, f)

        # saving lr scheduler and train state json
        if rank == 0:
            with open(LR_PATH, 'wb') as f:
                torch.save(lr_scheduler_state_dict, f)

            json_string = json.dumps(dataclasses.asdict(self.state), indent=2, sort_keys=True) + "\n"
            with open(TRAINER_STATE_NAME_PATH, 'w') as f:
                f.write(json_string)

        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
            "gpu": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        with open(RNG_PATH, 'wb') as f:
            torch.save(rng_states, f)

        logger.info(f"Checkpoint saved at {output_dir}")

    # def _save_checkpoint(self, model, trial, metrics=None):
    #     if getattr(self.args, 'tune_mm_mlp_adapter', False):
    #         from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
    #         checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

    #         run_dir = self._get_output_dir(trial=trial)
    #         output_dir = os.path.join(run_dir, checkpoint_folder)

    #         # Only save Adapter
    #         keys_to_match = ['mm_projector', 'vision_resampler']
    #         if getattr(self.args, "use_im_start_end", False):
    #             keys_to_match.extend(['embed_tokens', 'embed_in'])

    #         weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

    #         if self.args.local_rank == 0 or self.args.local_rank == -1:
    #             self.model.config.save_pretrained(output_dir)
    #             torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
    #     else:
    #         super(CambrianTrainer, self)._save_checkpoint(model, trial, metrics)
    """
    def get_train_dataloader(self) -> DataLoader:
        out = super().get_train_dataloader()
        return out.dataset
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
    
        # Initialize the distributed process group if not already initialized
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', init_method='env://')

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        ckpt_prefix = os.path.join(output_dir, "model_ckpt")
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        ckpt_path = f'{ckpt_prefix}_rank-{rank:08d}-of-{world_size:08d}.pth'

        if state_dict is None:
            state_dict = self.model.state_dict()

        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }

        torch.save(cpu_state_dict, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

        # Destroy the process group if no longer needed
        dist.destroy_process_group()

        """Override to add custom logs"""

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_tensor = torch.tensor(tr_loss, device=model.device)
            dist.all_reduce(tr_loss_tensor, op=dist.ReduceOp.SUM)
            tr_loss_scalar = tr_loss_tensor.item() / dist.get_world_size()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            # Add custom logs
            if self.args.unfreeze_mm_vision_tower:
                logs["mm_vision_tower_lr"] = self.optimizer.param_groups[2]['lr']

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
