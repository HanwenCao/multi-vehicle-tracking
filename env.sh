pip install /data/caohw9/cu100/torch-1.4.0+cu100-cp36-cp36m-linux_x86_64.whl
pip install /data/caohw9/cu100/torchvision-0.5.0+cu100-cp36-cp36m-linux_x86_64.whl
pip install 'moviepy<1.0.0'
pip install gputil pycocotools av
pip install /data/caohw9/cu100/detectron2-0.2+cu100-cp36-cp36m-linux_x86_64.whl

apt update && apt install -y libsm6 libxext6 libxrender-dev
apt-get install software-properties-common -y
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update
apt-get install gcc-4.9 -y
apt-get install --only-upgrade libstdc++6 -y