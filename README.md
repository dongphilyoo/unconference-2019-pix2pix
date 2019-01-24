# unconference-2019-pix2pix
ITP 2019 Unconference Pix2Pix Workshop

### [Pix2Pix](#Pix2Pix)
### [Data Preparation](#Create-own-dataset)
### [Training Pix2Pix](#Training)
### [Use the Model](#Use-the-Model-with-ml5.js)

## Pix2Pix
### What is it?
### What it does?
#### [Demo](https://dongphilyoo.github.io/pix2pix-ml5-demo/index.html)
## Create own dataset
![](https://cdn-images-1.medium.com/max/1600/1*QNZUc16K5Ooo8ZF0jaJJkQ.png)<br/>
It should consist of:<br/>
* 512x256 size (pairs of 256x256 images)
* hundreds of images (in same format)
* identical direction (A to B || B to A)

### Tools
[instagram-scraper](https://github.com/rarcega/instagram-scraper)<br/>
[body-pix / person segmentation](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)
#### Download dataset

## Training
### Prerequisite
#### Install Python3
```
# check python3 version
python3 --version

# install xcode
xcode-select --install

# install homebrew
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# check homebrew
brew doctor

# install python3 with homebrew
brew install python3

# check python3 version
python3 --version
```
#### Install pip
```
python --version
sudo easy_install pip
```
#### Install Tensorflow
```
pip install tensorflow

# make sure tensorflow installed
python -c "import tensorflow; print(tensorflow.__version__)"
```
#### Install Spell CLI
```
pip install spell

spell login
spell whoami
```
### Clone the repo
```
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
cd pix2pix-tensorflow
open .
```
#### Download dataset
[download](https://drive.google.com/drive/folders/1q_1yrHXaORVtu-9j2XXMviSRSK5NEraJ?usp=sharing)(24MB, 277 images)<br/>
```
# move downloaded folder to pix2pix-tensorflow folder

# Let spell know the newly downloaded dataset
git add .
git commit -m "added input image data"
```
### Train the model
```
# train the model

# If you add `--ngf 32 --ndf 32` when training the model, the output model will be smaller, and it will take less time to train.
# The command reduces the number of filters on generator and discriminator from 64(default) to 32.

spell run --machine-type V100 \
          --framework tensorflow \
          "python pix2pix.py \
          --mode train \
          --input_dir input \
          --output_dir output \
          --max_epochs 50 \
          --which_direction BtoA \
          --ngf 32 --ndf 32"

# After the run is finished (~5min)
# check out the result 
spell ls runs/YOUR_RUN_NUMBER
spell ls runs/YOUR_RUN_NUMBER/output
```
### Test the model
download test_images, locate into pix2pix-tensorflow root folder<br/>
[download test images](https://drive.google.com/open?id=18nqpuMwmTJukUijx_Zi9d-tuM2NoeEGM)
```
# in pix2pix-tensorflow root folder
mkdir output
cd output

# Copy the result folder, takes ~5min
spell cp runs/YOUR_RUN_NUMBER/output

# test the model
cd ..
python pix2pix.py \
     --mode test \
     --output_dir model_test \
     --input_dir test_images \
     --checkpoint checkpoint

# After testing, you should be able to see output images in the facades_test folder by opening the `index.html` file in the browser
```
### 🕒💸
![](https://i.imgur.com/wUknw0X.jpg)

## Demo:
Edges2Pikachu: [https://yining1023.github.io/pix2pix_spell/edges2pikachu/](https://yining1023.github.io/pix2pix_spell/edges2pikachu/)

## Training a Edges2Pikachu model: see instructions [https://github.com/yining1023/pix2pix_tensorflowjs_lite](https://github.com/yining1023/pix2pix_tensorflowjs_lite)

## Training a Label2Facades model with [Spell](http://spell.run)


```
# make sure you have Tensorflow 0.12.1 installed first
python -c "import tensorflow; print(tensorflow.__version__)"

# clone the repo
git clone https://github.com/affinelayer/pix2pix-tensorflow.git
cd pix2pix-tensorflow

# download the CMP Facades dataset http://cmp.felk.cvut.cz/~tylecr1/facade/
python tools/download-dataset.py facades

# Let spell know the newly downloaded dataset
git add .
git commit -m "added facade images"

# train the model
# this may take 1-9 hours depending on GPU, on CPU you will be waiting for a bit
# You could choose V100 or K80 as machine type
# If you add `--ngf 32 --ndf 32` when training the model: python pix2pix.py --mode train --output_dir pikachu_train --max_epochs 200 --input_dir pikachu/train --which_direction BtoA --ngf 32 --ndf 32, the model will be smaller 13.6 MB, and it will take less time to train.

spell run --machine-type V100 \
            --framework tensorflow \
  "python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA"

# After the run is finished (1h, 8m, 16s)
# check out the result 
spell ls runs/YOUR_RUN_NUMBER
spell ls runs/YOUR_RUN_NUMBER/facades_train

# Go to the pix2pix-tensorflow repo
cd pix2pix-tensorflow/
mkdir facades_train
cd facades_train

# Copy the result folder,  takes ~15mins
spell cp runs/YOUR_RUN_NUMBER/facades_train

# test the model
cd ..
python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir facades/val \
  --checkpoint facades_train
# After testing, you should be able to see output images in the facades_test folder by opening the `index.html` file in the browser

# export the model
python pix2pix.py --mode export --output_dir export/ --checkpoint facades_train/ --which_direction BtoA
# It will create a new export folder

# Port the model to tensorflow.js (python 2 doesn’t have tempfile, so use python3 instead)
cd server
cd static
mkdir models
cd ..
python3 tools/export-checkpoint.py --checkpoint ../export --output_file static/models/facades_BtoA.pict
# We should be able to get a file named facades_BtoA.pict, which is 54 MB.
# If you add `--ngf 32 --ndf 32` when training the model: python pix2pix.py --mode train --output_dir pikachu_train --max_epochs 200 --input_dir pikachu/train --which_direction BtoA --ngf 32 --ndf 32, the model will be smaller 13.6 MB, and it will take less time to train.

# Copy the model we got to the `models` folder.

```

# Create own dataset



pix2pix hd hi-res
gene
yinying
pix2pix tensorflow
ml5
medium
instagram

