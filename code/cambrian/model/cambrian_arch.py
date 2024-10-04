#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from ezcolorlog import root_logger as logger

from .multimodal_encoder.builder import build_vision_tower_aux_list
from .multimodal_projector.builder import build_vision_projector
from .vision_sampler import VisionTokenSampler

from cambrian.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN



class CambrianMetaModel:

    def __init__(self, config):
        super(CambrianMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):

            projector_type = getattr(config, 'mm_projector_type', 'linear')
            if projector_type == 'sva':

                vision_hidden_size = config.vision_hidden_size
                num_query_group = config.num_query_group
                query_num_list = config.query_num_list
                connector_only = config.connector_only
                connector_depth = config.connector_depth
                self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True)
                self.mm_projector = nn.Sequential(nn.Linear(vision_hidden_size*num_query_group, config.hidden_size), nn.GELU(), nn.Linear(config.hidden_size, config.hidden_size))

                image_token_len = config.image_token_len
                vision_tower_aux_token_len_list = self.config.mm_vision_tower_aux_token_len_list
                cross_att_token_len_list = [int(vision_tower_aux_token_len**0.5) // int(image_token_len**0.5) for vision_tower_aux_token_len in vision_tower_aux_token_len_list]

                for aux_i, vision_tower_aux in enumerate(self.vision_tower_aux_list):
                    setattr(self, 'mm_projector_aux_{}'.format(aux_i), nn.Sequential(nn.Linear(vision_tower_aux.hidden_size, vision_hidden_size), nn.GELU(), nn.Linear(vision_hidden_size, vision_hidden_size), nn.LayerNorm(vision_hidden_size)))

                for query_group_i in range(num_query_group):
                    cross_att_token_len_list = [int(vision_tower_aux_token_len**0.5) // int(query_num_list[query_group_i]**0.5) for vision_tower_aux_token_len in vision_tower_aux_token_len_list]
                    setattr(self, "vision_sampler_{}".format(query_group_i), VisionTokenSampler(vision_hidden_size, vision_hidden_size, [vision_hidden_size]*len(self.vision_tower_aux_list), cross_att_token_len_list, vision_hidden_size, connector_depth))

                if not connector_only:
                    num_of_vision_sampler_layers = config.num_of_vision_sampler_layers = config.num_of_vision_sampler_layers
                    config.start_of_vision_sampler_layers = config.start_of_vision_sampler_layers
                    config.stride_of_vision_sampler_layers = config.stride_of_vision_sampler_layers
                    cross_att_token_len_list = [int(vision_tower_aux_token_len**0.5) // int(image_token_len**0.5) for vision_tower_aux_token_len in vision_tower_aux_token_len_list]
                    self.vision_sampler_layers = nn.ModuleList(
                    [VisionTokenSampler(config.hidden_size, vision_hidden_size, [vision_hidden_size]*len(self.vision_tower_aux_list), cross_att_token_len_list, vision_hidden_size, 1) for layer_idx in range(0, num_of_vision_sampler_layers)]
                    )


                self.vision_query = nn.Parameter(
                    torch.randn((num_query_group, vision_hidden_size), dtype=self.dtype)
                )


                self.image_newline = nn.Parameter(
                        torch.empty(config.hidden_size, dtype=self.dtype)
                    )

            else:
                self.vision_tower_aux_list = build_vision_tower_aux_list(config, delay_load=True)
                config.mm_hidden_size = sum([vision_tower_aux.hidden_size for vision_tower_aux in self.vision_tower_aux_list]) 
                self.mm_projector = build_vision_projector(config)
                self.image_newline = nn.Parameter(
                        torch.empty(config.hidden_size, dtype=self.dtype)
                    )

    # def get_vision_tower(self):
    #     vision_tower = getattr(self, 'vision_tower', None)
    #     if type(vision_tower) is list:
    #         vision_tower = vision_tower[0]
    #     return vision_tower

    def get_vision_tower_aux_list(self):
        vision_tower_aux_list = getattr(self, 'vision_tower_aux_list', None)
        return vision_tower_aux_list

    def initialize_vision_modules(self, model_args, fsdp=None):
        # vision_tower = model_args.vision_tower
        num_query_group = model_args.num_query_group
        query_num_list = model_args.query_num_list
        vision_hidden_size = model_args.vision_hidden_size
        vision_tower_aux_list = model_args.vision_tower_aux_list
        vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        image_token_len = model_args.image_token_len
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        connector_only = model_args.connector_only
        connector_depth = model_args.connector_depth

        # self.config.mm_vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        self.config.num_query_group = num_query_group
        self.config.query_num_list = query_num_list
        assert num_query_group == len(query_num_list)
        self.config.connector_depth = connector_depth
        self.config.mm_vision_tower_aux_list = vision_tower_aux_list
        self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list
        self.config.connector_only = connector_only

        if self.get_vision_tower_aux_list() is None:
            vision_tower_aux_list = build_vision_tower_aux_list(model_args)
            if model_args.unfreeze_mm_vision_tower:
                self.vision_tower_aux_list = nn.ModuleList(vision_tower_aux_list)
            else:
                self.vision_tower_aux_list = vision_tower_aux_list
        else:
            vision_tower_aux_list = self.vision_tower_aux_list
            for vision_tower_aux in vision_tower_aux_list:
                vision_tower_aux.load_model()
            
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.vision_hidden_size = vision_hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:

            if self.config.mm_projector_type == 'sva':
                self.mm_projector = nn.Sequential(nn.Linear(vision_hidden_size*num_query_group, self.config.hidden_size), nn.GELU(), nn.Linear(self.config.hidden_size, self.config.hidden_size))
                for aux_i, vision_tower_aux in enumerate(vision_tower_aux_list):
                    setattr(self, 'mm_projector_aux_{}'.format(aux_i), nn.Sequential(nn.Linear(vision_tower_aux.hidden_size, vision_hidden_size), nn.GELU(), nn.Linear(vision_hidden_size, vision_hidden_size), nn.LayerNorm(vision_hidden_size)))

                # vision sampler for each group of query as the connector before the LLM
                for query_group_i in range(num_query_group):
                    cross_att_token_len_list = [int(vision_tower_aux_token_len**0.5) // int(query_num_list[query_group_i]**0.5) for vision_tower_aux_token_len in vision_tower_aux_token_len_list]
                    setattr(self, "vision_sampler_{}".format(query_group_i), VisionTokenSampler(vision_hidden_size, vision_hidden_size, [vision_hidden_size]*len(vision_tower_aux_list), cross_att_token_len_list, vision_hidden_size, connector_depth))

                # sampler layers within LLM
                if not connector_only:
                    num_of_vision_sampler_layers = self.config.num_of_vision_sampler_layers = model_args.num_of_vision_sampler_layers
                    self.config.start_of_vision_sampler_layers = model_args.start_of_vision_sampler_layers
                    self.config.stride_of_vision_sampler_layers = model_args.stride_of_vision_sampler_layers
                    cross_att_token_len_list = [int(vision_tower_aux_token_len**0.5) // int(image_token_len**0.5) for vision_tower_aux_token_len in vision_tower_aux_token_len_list]
                    self.vision_sampler_layers = nn.ModuleList(
                    [VisionTokenSampler(self.config.hidden_size, vision_hidden_size, [vision_hidden_size]*len(vision_tower_aux_list), cross_att_token_len_list, vision_hidden_size, 1) for layer_idx in range(0, num_of_vision_sampler_layers)]
                    )
                vision_embed_std = 1 / torch.sqrt(torch.tensor(vision_hidden_size, dtype=self.dtype))
                self.vision_query = nn.Parameter(
                    torch.randn((num_query_group, vision_hidden_size), dtype=self.dtype) * vision_embed_std
                )
                
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

            else:
                self.config.mm_hidden_size = sum([vision_tower_aux.hidden_size for vision_tower_aux in vision_tower_aux_list]) 
                self.mm_projector = build_vision_projector(self.config)
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True
        print("ac2")
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword+'.' in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'),strict=True)

            if self.config.mm_projector_type == 'sva':
                for aux_i in range(len(vision_tower_aux_list)):
                    getattr(self, 'mm_projector_aux_{}'.format(aux_i)).load_state_dict(get_w(mm_projector_weights, 'mm_projector_aux_{}'.format(aux_i)),strict=True)

                for query_group_i in range(num_query_group):
                    getattr(self, "vision_sampler_{}".format(query_group_i)).load_state_dict(get_w(mm_projector_weights, "vision_sampler_{}".format(query_group_i)),strict=True)

                if not connector_only:
                    self.vision_sampler_layers.load_state_dict(get_w(mm_projector_weights, 'vision_sampler_layers'),strict=True)
                self.vision_query.data = mm_projector_weights['model.vision_query']
            self.image_newline.data = mm_projector_weights['model.image_newline']


def unmask_attention_mask(mask, original_size):
    original_w, original_h = original_size
    cur_h, cur_w = mask.shape[1:3]

    original_aspect_ratio = original_w / original_h
    current_aspect_ratio = cur_w / cur_h

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = cur_w / original_w
        new_height = int(original_h * scale_factor)
        padding = (cur_h - new_height) // 2
        if padding > 0:
            mask[:, :padding, :]=0
            mask[:, -padding:, :]=0
        return mask
    else:
        scale_factor = cur_h / original_h
        new_width = int(original_w * scale_factor)
        padding = (cur_w - new_width) // 2
        if padding > 0:
            mask[:, :, :padding]=0
            mask[:, :, -padding:]=0
        return mask


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:3]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class CambrianMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    # def get_vision_tower(self):
    #     return self.get_model().get_vision_tower()

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def rearrange_vision_tower_features_train(self, vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list, query_side_len):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature, vision_tower_aux_attention_masks in zip(vision_tower_aux_feature_list, vision_tower_aux_attention_masks_list):
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1]**0.5)
            assert (aux_height//query_side_len) * query_side_len == aux_height

            reduce_factor = (aux_height//query_side_len)
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature.view(bs, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
            vision_tower_aux_feature_rearranged = vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5).contiguous().flatten(0,2).flatten(1,2)

            vision_tower_aux_attention_masks_rearranged = vision_tower_aux_attention_masks.view(bs*query_side_len*query_side_len, reduce_factor*reduce_factor)

            vision_tower_aux_feature_rearranged_list.append(vision_tower_aux_feature_rearranged)
            vision_tower_aux_attention_masks_rearranged_list.append(vision_tower_aux_attention_masks_rearranged)
        return vision_tower_aux_feature_rearranged_list, vision_tower_aux_attention_masks_rearranged_list

    def rearrange_vision_tower_features_inference(self, vision_tower_aux_feature_list, query_side_len, image_sizes, unpad=False):
        vision_tower_aux_feature_rearranged_list = []
        vision_tower_aux_attention_masks_rearranged_list = []
        bs = vision_tower_aux_feature_list[0].shape[0]
        for vision_tower_aux_feature in vision_tower_aux_feature_list:
            aux_height = aux_width = int(vision_tower_aux_feature.shape[1]**0.5)
            assert (aux_height//query_side_len) * query_side_len == aux_height

            reduce_factor = (aux_height//query_side_len)

            vision_tower_aux_feature_rearranged = []
            vision_tower_aux_attention_masks_rearranged = []
            for batch_i in range(bs):
                image_size = image_sizes[batch_i]
                cur_vision_tower_aux_feature = vision_tower_aux_feature[batch_i]

                cur_vision_tower_aux_attention_masks_rearranged = torch.ones((1, aux_height, aux_width), dtype=torch.bool, device=cur_vision_tower_aux_feature.device)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature.view(1, query_side_len, reduce_factor, query_side_len, reduce_factor, -1)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature_rearranged.permute(0, 1, 3, 2, 4, 5).contiguous()
                if unpad:
                    cur_vision_tower_aux_feature_rearranged = unpad_image(cur_vision_tower_aux_feature_rearranged, image_size)
                cur_vision_tower_aux_feature_rearranged = cur_vision_tower_aux_feature_rearranged.flatten(0,2).flatten(1,2) # query_side_len*query_side_len X reduce_factor*reduce_factor X C

                cur_vision_tower_aux_attention_masks_rearranged = unmask_attention_mask(cur_vision_tower_aux_attention_masks_rearranged, image_size)
                cur_vision_tower_aux_attention_masks_rearranged = cur_vision_tower_aux_attention_masks_rearranged.view(1, query_side_len, reduce_factor, query_side_len, reduce_factor).permute(0, 1, 3, 2, 4).contiguous()
                if unpad:
                    cur_vision_tower_aux_attention_masks_rearranged = unpad_image(cur_vision_tower_aux_attention_masks_rearranged, image_size)
                cur_vision_tower_aux_attention_masks_rearranged = cur_vision_tower_aux_attention_masks_rearranged.flatten(0,2).flatten(1,2)

                cur_vision_tower_aux_attention_masks_rearranged[cur_vision_tower_aux_attention_masks_rearranged.sum(-1)==0] = True

                vision_tower_aux_feature_rearranged.append(cur_vision_tower_aux_feature_rearranged)
                vision_tower_aux_attention_masks_rearranged.append(cur_vision_tower_aux_attention_masks_rearranged)

            vision_tower_aux_feature_rearranged = torch.cat(vision_tower_aux_feature_rearranged, 0)
            vision_tower_aux_attention_masks_rearranged = torch.cat(vision_tower_aux_attention_masks_rearranged, 0)


            vision_tower_aux_feature_rearranged_list.append(vision_tower_aux_feature_rearranged)
            vision_tower_aux_attention_masks_rearranged_list.append(vision_tower_aux_attention_masks_rearranged)

        return vision_tower_aux_feature_rearranged_list, vision_tower_aux_attention_masks_rearranged_list

    def encode_images(self, image_aux_list):
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        image_aux_features_list = []
        for image_aux, vision_tower_aux in zip(image_aux_list, vision_tower_aux_list):
            image_aux_features = vision_tower_aux(image_aux)
            image_aux_features_list.append(image_aux_features)
        return image_aux_features_list

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_aux_attention_masks_list=None, image_sizes=None
    ):
        # vision_tower = self.get_vision_tower()
        vision_tower_aux_list = self.get_model().get_vision_tower_aux_list()
        if vision_tower_aux_list is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None, None, None

        image_aux_list = images

        bs = image_aux_list[0].shape[0]
        dtype = image_aux_list[0].dtype

        image_token_len = self.get_model().config.image_token_len
        query_num_list = self.get_model().config.query_num_list

        final_height = final_width  = int(image_token_len**0.5)

        final_image_features_list = []

        # only needed for sva
        vision_tower_aux_feature_list_final = None
        vision_tower_aux_attention_masks_list_final = None
        global_context_feature_final = None

        image_aux_features_list = self.encode_images(image_aux_list)

        if self.get_model().config.mm_projector_type == 'sva':
            vision_tower_aux_feature_list = []
            vision_tower_aux_attention_masks_list = []
            # get vision tokens from each vision tower
            for aux_i in range(len(vision_tower_aux_list)):
                image_aux_features = image_aux_features_list[aux_i]

                image_aux_features = getattr(self.get_model(), 'mm_projector_aux_{}'.format(aux_i))(image_aux_features).to(dtype)
                if aux_i == 0:
                    global_context_feature = image_aux_features.mean(1).view(bs, 1, 1, -1)

                vision_tower_aux_feature_list.append(image_aux_features)

            # perform vision sampling for each query group
            for query_group_i, query_num in enumerate(query_num_list):
                query_features_i = self.get_model().vision_query[query_group_i, :].view(1, 1, 1, -1).expand(bs, query_num, -1, -1)
                global_context_feature_i = global_context_feature.expand(-1, query_num, 1, -1).flatten(0,1)
                query_side_len = int(query_num**0.5)
                vision_tower_aux_feature_list_i, vision_tower_aux_attention_masks_list_i = self.rearrange_vision_tower_features_train(vision_tower_aux_feature_list, image_aux_attention_masks_list, query_side_len)

                query_features_i = getattr(self.get_model(), "vision_sampler_{}".format(query_group_i))(query_features_i.flatten(0,1), global_context_feature_i, *vision_tower_aux_feature_list_i, *vision_tower_aux_attention_masks_list_i)
                query_features_i = query_features_i.view(bs, query_num, -1)
                # interpolate to the final target size
                if query_side_len != final_height:
                    query_features_i = query_features_i.permute(0, 2, 1).contiguous().view(bs, -1, query_side_len, query_side_len)
                    query_features_i = F.interpolate(query_features_i.float(), 
                                                    size=(final_height, final_width), 
                                                    mode='bilinear', 
                                                    align_corners=False).to(dtype=query_features_i.dtype)
                    query_features_i = query_features_i.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                final_image_features_list.append(query_features_i)

            vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final = self.rearrange_vision_tower_features_train(vision_tower_aux_feature_list, image_aux_attention_masks_list, final_height)
            global_context_feature_final = global_context_feature.expand(-1, final_height*final_width, 1, -1).flatten(0,1)
        else:
            final_image_features_list = image_aux_features_list

        image_features = torch.cat(final_image_features_list, -1)
        image_features = self.get_model().mm_projector(image_features).to(dtype)

        image_features = image_features.view(image_features.shape[0], final_height, final_width, -1)
        image_features = torch.cat((
            image_features,
            self.model.image_newline[None, None, None, :].expand(image_features.shape[0], final_height, 1, -1)
        ), dim=2)
        image_features = image_features.flatten(1, 2)
        final_size = [(final_height, final_width)]*bs
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        if True:

            # embed the input_ids
            new_input_ids_padded_for_emb = torch.where(input_ids==IMAGE_TOKEN_INDEX, 0, input_ids)
            input_embeds = self.get_model().embed_tokens(new_input_ids_padded_for_emb)
            new_input_embeds = []
            cur_image_idx = 0
            # insert the image embeddings
            for batch_idx, (cur_input_embeds, cur_input_ids) in enumerate(zip(input_embeds, input_ids)):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    cur_image_idx += 1
                    new_input_embeds.append(cur_input_embeds)
                    continue

                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]

                cur_input_embeds_im_replaced = []

                prev_image_length = 0
                for i in range(len(image_token_indices) - 1):
                    # skip the image tokens (1 indicator + (image_length-1) paddings)
                    cur_input_embeds_im_replaced.append(cur_input_embeds[image_token_indices[i]+1+prev_image_length:image_token_indices[i+1]])
                    if i < len(image_token_indices) - 2:
                        cur_image_features = image_features[cur_image_idx]
                        prev_image_length = len(cur_image_features)-1
                        cur_image_idx += 1
                        cur_input_embeds_im_replaced.append(cur_image_features)

                cur_input_embeds_im_replaced = [x.to(self.device) for x in cur_input_embeds_im_replaced]
                new_input_embeds.append(torch.cat(cur_input_embeds_im_replaced))

            new_input_embeds = torch.stack(new_input_embeds)
            return None, position_ids, attention_mask, past_key_values, new_input_embeds, labels, vision_tower_aux_feature_list_final, vision_tower_aux_attention_masks_list_final, final_size, global_context_feature_final

        
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
