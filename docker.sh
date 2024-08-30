git clone https://github.com/believewhat/PMC-Cambrian.git
cd ./code/cambrian
conda create -n pmc_cambrian python=3.10 -y
conda activate pmc_cambrian
pip install --upgrade pip  # enable PEP 660 support
pip install -e ".[gpu]"
pip install torch~=2.2.0 -f https://storage.googleapis.com/libtpu-releases/index.html
