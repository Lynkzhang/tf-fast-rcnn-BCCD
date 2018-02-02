# Faster-RCNN for BCCD

[Original README.md](./origin.md)

Below is Faster-RCNN for BCCD:

```shell
git clone https://github.com/Shenggan/tf-fast-rcnn-BCCD.git
cd tf-fast-rcnn-BCCD

pip install -r requirments.txt

make
cd ..

# install coco python api
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..

# download resnet init weights
mkdir -p data/imagenet_weights
cd data/imagenet_weights
wget -v http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
tar -xzvf resnet_v1_101_2016_08_28.tar.gz
mv resnet_v1_101.ckpt res101.ckpt
cd ../..

# download BCCD dataset
cd data
wget http://59.78.0.210:8123/VOCdevkit2007.zip
unzip VOCdevkit2007.zip
cd ..

# train
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101
# test
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc res101

# test with img
python tools/demo2.py
```

