# unconference-2019-pix2pix
ITP 2019 Unconference Pix2Pix Workshop

### [Pix2Pix](#Pix2Pix)
### [Data Preparation](#Create-own-dataset)
### [Training Pix2Pix](#Training)
### [Use the Model](#Use-the-Model-with-ml5.js)

# Pix2Pix
## What is it?
## What it does?

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
