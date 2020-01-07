# TODO: create shell script for running your YoloV1-vgg16bn model
#!/bin/bash
echo "-------------data downloading-------------"
wget -O hw2-175.pth 'https://www.dropbox.com/s/027k6kikf1agu8a/hw2-175.pth?dl=1'
#mv ./hw2-175.pth?dl=1 ./hw2-175.pth
echo "-----------finish downloading-------------"
python3 testBase.py $1 $2