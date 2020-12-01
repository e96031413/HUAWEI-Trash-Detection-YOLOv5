# HUAWEI-Trash-Detection-YOLOv5
Training dataset from Huawei Cloud competition 2020

### Download the original dataset from Baidu Yun
link: https://pan.baidu.com/s/1lh1D1wvXUV3rjJOUpWsBlA password: znk3

### Data format conversion

We use dataset from Huawei Cloud competition 2020

The original format is VOC2007, we use [convert2Yolo](https://github.com/ssaru/convert2Yolo) to convert to YOLO supported format with the following command:

```
python3 example.py --datasets VOC --img_path ../trainval/VOC2007/JPEGImages --label ../trainval/VOC2007/Annotations/ --convert_output_path ./yolo/ --img_type ".jpg" --manifest_path ./ --cls_list_file ../trainval/train_classes.txt
```
note: you should change to path according to your own folder.

### Split training set / val set / test set
```
# copy all your img and labels(coco format) to huawei-tmp
mkdir ./huawei-tmp/         

# split-folders
pip3 install split-folders

#current dir should be ./ instead of ./huawei-tmp

python3
import splitfolders
splitfolders.ratio('huawei-tmp', output="huawei-trash-dataset", seed=1337, ratio=(.8, 0.1,0.1)) 
```

To be continued.......
