# HUAWEI-Trash-Detection-YOLOv5
Training dataset from Huawei Cloud competition 2020

### Download the original dataset from BaiduYun
The dataset source is from [this post](https://blog.csdn.net/qq_38410428/article/details/106974147)

link: https://pan.baidu.com/s/1lh1D1wvXUV3rjJOUpWsBlA 

password: znk3

### Data format conversion

We use dataset from Huawei Cloud competition 2020

The original format is VOC2007, we use [convert2Yolo](https://github.com/ssaru/convert2Yolo) to convert to YOLO supported format with the following command:

```
python3 example.py --datasets VOC --img_path ../trainval/VOC2007/JPEGImages --label ../trainval/VOC2007/Annotations/ --convert_output_path ./yolo/ --img_type ".jpg" --manifest_path ./ --cls_list_file ../trainval/train_classes.txt
```
note: you should change the path according to your own folder.

### Split training set / val set / test set
put all your img and labels(coco format) to **huawei-tmp/images/** and **huawei-tmp/labels/** respectively.
```
mkdir ./huawei-tmp/images
mkdir ./huawei-tmp/labels


# split-folders
pip3 install split-folders

# current dir should be ./ instead of ./huawei-tmp

# you can change to any split (0.6, 0.2, 0.2) for 60% : 20%: 20%
python3
import splitfolders
splitfolders.ratio('huawei-tmp', output="huawei-trash-dataset", seed=1337, ratio=(.8, 0.1,0.1)) 
```

### YOLOv5 Training
Assuming you have setted up the environment,we provide pre-configured [trash.yaml](https://github.com/e96031413/HUAWEI-Trash-Detection-YOLOv5/blob/main/trash.yaml), you can put it in yolov5/data/trash.yaml(change path at your own) and train with the following command (2GPUs):
```
cd yolov5
python -m torch.distributed.launch --nproc_per_node 2 train.py --cfg models/yolov5s.yaml --img 640 --epochs 100 --batch-size 16 --data trash.yaml --weights 'yolov5s.pt' --devices 2,3
```
To be continued.......
