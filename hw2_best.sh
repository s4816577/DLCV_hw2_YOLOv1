# TODO: create shell script for running your improved model
#!/bin/bash
echo "-------------data downloading-------------"
wget -O hw2-97.pth 'https://www.dropbox.com/s/qhfaqs9jxd64k9j/hw2-97.pth?dl=1'
#mv ./hw2-97.pth?dl=1 ./hw2-97.pth
echo "-----------finish downloading-------------"
python3 testImproved.py $1 $2