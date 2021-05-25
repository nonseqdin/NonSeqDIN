mkdir data
mkdir data/taobao_data/
unzip UserBehavior.csv.zip
mv UserBehavior.csv ./data/taobao_data/

python preprocess/taobao_prepare_1.py
python preprocess/to_odps_taobao_1.py