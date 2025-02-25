# Prepare OpenGF_cls dataset
## download 
Download the Train, Validation, and Test files of [OpenGF dataset](https://github.com/Nathan-UW/OpenGF).

Put the downloaded folders  (or their symlinks) to ALS_pretraining/data/opengfcls

## sub-divide the tiles 
```
python script/sliding_split_las_from_text.py
```

## create info file
```
create_info.py -i .
```
For detailed explanations of info file please refer to [OpenPCDet v0.6.0](https://github.com/open-mmlab/OpenPCDet). 
