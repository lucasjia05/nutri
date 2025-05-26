source $HOME/miniconda3/etc/profile.d/conda.sh
conda create -n qlora2 python=3.12 -y
conda activate qlora2

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip3 install -r requirements.txt

pip install --force-reinstall -v "triton==3.1.0"
