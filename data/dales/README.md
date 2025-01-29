# Prepare DALES dataset
## download 
```
cd script
sh download.sh
```
Extract the downloaded archive file and put the donwloaded train and test folders (or their symlinks) to ALS_pretraining/data/dales/

## create the validation file
```
python3 sliding_split_las.py
```

## create info file
```
python3 create_info.py -i ../
```
For detailed explanations of info file please refer to [OpenPCDet v0.6.0](https://github.com/open-mmlab/OpenPCDet). 

