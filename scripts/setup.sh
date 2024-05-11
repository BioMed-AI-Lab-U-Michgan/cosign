git clone https://github.com/VinAIResearch/blur-kernel-space-exploring bkse

# Download VAE kernel for nonlinear deblur
# If the command failed, it means that the link has been expired
# In this case, please download from https://github.com/VinAIResearch/blur-kernel-space-exploring?tab=readme-ov-file#model-zoo on your own
cd bkse/experiments/pretrained
filename='GOPRO_wVAE.pth'
fileid='1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy'
wget --load-cookies /tmp/cookies.txt "https://drive.usercontent.google.com/download?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.usercontent.google.com/download?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

cd ../../..
conda create -n cosign python=3.9.12 -y
conda activate cosign
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -e .
