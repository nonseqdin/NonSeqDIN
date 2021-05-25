
mkdir -p data/alipay_data/
unzip IJCAI16_data.zip
mv ijcai2016_taobao.csv ./data/alipay_data/

python preprocess/alipay_prepare_1.py
python preprocess/to_odps_alipay_1.py
