## Convolutional Neural Network
  Реализация сверточной сети с использованием Theano. Развитие кода, описанного в [книге](http://neuralnetworksanddeeplearning.com/)

## Зависимости
- python3 (numpy, matplotlib)
- Theano 0.8.2
- CUDA (опционально)

Для обучения сети лучше использовать GPU, в моем случае это дало почти пятикратный прирост производительности,
но установка CUDA не тривиальна.  
Для использования уже обученной сети достаточно и CPU (Шаг I).

## Theano and GPU support
### Warning. This is not the instruction, this is that I do to get GPU support

My hardware and software.  
NVIDIA GT 940M, nvidia-367 driver.  
ubuntu-gnome 16.04 with gdm.  

based on this manual http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu  
and this https://medium.com/@wchccy/install-theano-cuda-8-in-ubuntu-16-04-bdb02773e1ea


## I. Theano basic

> sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git  
> sudo pip install Theano  

## II. GPU support
To check instalation - run gpu_test.py  

### 1. CUDA Toolkit
   I install version 8.0 and use runfile (local), [link](https://developer.nvidia.com/cuda-downloads) to actual version.  

   Stop your dm, in my case - gdm.  
>   sudo service gdm stop  

   Open tty1 (ctrl + alt + f1) and run installer (better to install all).  
>   sudo sh cuda_installer.run  
>   reboot  

### 2. GCC version magic from tutorials
   Maybe you can skip this step and if after all Theano won't work, try to do this.  

### 3. Create ~/.theanorc
Or use THEANO_FLAGS environment variable.  

>	[global]  
>	device = gpu  
>	floatX = float32  
>
>	[cuda]  
>	root = /usr/local/cuda-8.0  
>
>	[nvcc]  
>	flags=-D_FORCE_INLINES  

### 4. Also add to ~/.bashrc this

>   export PATH="/usr/local/cuda-8.0/bin:$PATH"  
>   export CUDA_HOME="/usr/local/cuda:$CUDA_HOME"  
>   export LD_LIBRARY_PATH="/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH"  

   and execute this
>   sudo echo "/usr/local/cuda-8.0/lib64" > /etc/ld.so.conf.d/cuda.conf  
>   sudo ldconfig  
>   reboot  

## III. Troubleshooting
In case nvidia driver fault - reinstall it.  
In case dm fault - reconfigure it.  
And start again...
