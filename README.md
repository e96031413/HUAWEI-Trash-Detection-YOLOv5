# HUAWEI-Trash-Detection-YOLOv5
Training dataset from Huawei Cloud competition 2020 with YOLOv5

### Download the original dataset from BaiduYun
The dataset source is from [this post](https://blog.csdn.net/qq_38410428/article/details/106974147)

link: https://pan.baidu.com/s/1lh1D1wvXUV3rjJOUpWsBlA  ； password: znk3

or you can download from google drive [here](https://drive.google.com/file/d/1RlVXHu0ndlfYa1R_QPZusF_abp85bECr/view?usp=sharing)

### Download the post-process dataset from Google Drive(Data format conversion part)
You can download the processed dataset [here](https://drive.google.com/file/d/16VZwblyDb37hONzrfpVLwKfh3fcUeRw3/view?usp=sharing)

Tha maximum image size is 5.25 MB, the minimum image size is 1.25 KB

**train:val= 80 : 20**

-train: 9410 images

-val: 2349 images

### Dataset classes

```
'Disposable Fast Food Box','Book Paper','Power Bank','Leftovers','Package','Trash Can','Plastic Utensils','Plastic Toys',
'Plastic Hangers',"Big Bones","Dry Battery","Express Paper Bag", "Plug Wire", "Old Clothes", "Can", "Pillow",
"Peel and Pulp", "Stuffed Toy", "Defiled Plastic", "Contaminated paper","Toilet care products", "Cigarette butts",
"Toothpicks", "Glassware","Baffle", "Chopsticks", "Carton box", "Flower pot", "Tea residue", "Cai Bang Cai Ye",
"Egg Shell", "Sauce Bottle", "Ointment", "Expired Medicine", "Wine Bottle", "Metal Kitchenware", "Metal Utensils",
"Metal Food Cans", "Pots" , "Ceramic utensils", "shoes", "edible oil drums", "drink bottles", "fish bones"
```

### Dataset classes in Traditional Chinese
You can refer to [trash-中文.yaml](https://github.com/e96031413/HUAWEI-Trash-Detection-YOLOv5/blob/main/trash-%E4%B8%AD%E6%96%87.yaml)
but do not use it to train model, it could result garbled text on bounding box.
```
'一次性快餐盒', '書籍紙張', '充電寶', '剩飯剩菜', '包', '垃圾桶', '塑料器皿', '塑料玩具', '塑料衣架', '大骨頭', '乾電池',
'快遞紙袋', '插頭電線', '舊衣服', '易拉罐','枕頭', '果皮果肉', '毛絨玩具', '污損塑料', '污損用紙', '洗護用品', '煙蒂',
'牙簽', '玻璃器皿', '砧板', '筷子', '紙盒紙箱', '花盆', '茶葉渣', '菜幫菜葉','蛋殼', '調料瓶', '軟膏', '過期藥物',
'酒瓶', '金屬廚具', '金屬器皿', '金屬食品罐', '鍋', '陶瓷器皿', '鞋', '食用油桶', '飲料瓶', '魚骨'
```

### Data format conversion

We use dataset from Huawei Cloud competition 2020

The original format is VOC2007, we use [convert2Yolo](https://github.com/ssaru/convert2Yolo) to convert to YOLO supported format with the following command:

```
python3 example.py --datasets VOC --img_path ../trainval/VOC2007/JPEGImages --label ../trainval/VOC2007/Annotations/ --convert_output_path ./yolo/ --img_type ".jpg" --manifest_path ./ --cls_list_file ../trainval/train_classes.txt
```
note: you should change the path according to your own folder.

### Split training set / val set
put all your img and labels(coco format) to **huawei-tmp/images/** and **huawei-tmp/labels/** respectively.

and use [this script](https://gist.github.com/e96031413/7b2d832d1cc12a11be374b1c1a570aa9#file-makedataset-py) to split dataset(change to your own path)

### YOLOv5 Training
Assuming you have setted up the environment,we provide pre-configured [trash.yaml](https://github.com/e96031413/HUAWEI-Trash-Detection-YOLOv5/blob/main/trash.yaml), you can put it in yolov5/data/trash.yaml(change path at your own) and train with the following command (2GPUs):

#### with pre-trained weight
```
YOLOv5s:
python -m torch.distributed.launch --nproc_per_node 2 train.py --cfg models/yolov5s.yaml --img 640 --epochs 100 --batch-size 64 --data trash.yaml --weights 'yolov5s.pt' --devices 2,3

YOLOv4-tiny:
./darknet detector train cfg/huawei-trash.data cfg/yolov4-tiny-huawei.cfg yolov4-tiny.conv.29 -gpus 0 -dont_show -map |tee -a v4-tiny-huawei-dataset.txt

YOLOv3-tiny:
./darknet detector train cfg/huawei-trash.data cfg/yolov3-tiny-huawei.cfg yolov3-tiny.conv.15 -gpus 0 -dont_show -map |tee -a v3-tiny-huawei-dataset.txt
```
#### Testing the trained weight
```
python test.py --weights runs/train/exp3/weights/best.pt --data trash.yaml --img 640
```
#### Training result
```
YOLOv5s, pre-trained weight: yolov5s.pt, 300 epochs, batch size = 64, img size = 640
Class    Images     Targets    P       R      F1     mAP@.5  mAP@.5:.95: 100%|██████████████| 75/75 [00:16<00:00,  4.56it/s]
 all    2.38e+03    3.91e+03   0.556   0.697  0.618  0.663   0.506

YOLOv5s, pre-trained weight: yolov5s.pt, 300 epochs, batch size = 64, img size = 416
Class    Images     Targets    P       R      F1     mAP@.5  mAP@.5:.95: 100%|██████████████| 75/75 [00:32<00:00,  2.33it/s]
  all    2.38e+03   3.91e+03   0.556   0.71   0.623  0.68    0.545

YOLOv4-tiny, pre-trained weight: yolov4-tiny.conv.29, max_batches=50000, batch size = 64, img size = 416
conf_thresh = 0.25, precision = 0.64, recall= 0.57, F1-score = 0.60
for conf_thresh = 0.25, TP = 2235, FP = 1267, FN = 1679, average IoU = 49.50 %
IoU threshold = 50 %, mean average precision (mAP@0.50) = 0.580316, or 58.03 % 

YOLOv3-tiny, pre-trained weight: yolov3-tiny.conv.15, max_batches=50000, batch size = 64, img size = 416
conf_thresh = 0.25, precision = 0.73, recall = 0.32, F1-score = 0.44 
for for conf_thresh = 0.25, TP = 1235, FP = 461, FN = 2679, average IoU = 55.84 % 
IoU threshold = 50 %, mean average precision (mAP@0.50) = 0.469817, or 46.98 %
```
|    Model    | mAP@0.5 | AVG FPS |
|:-----------:|:-------:|:-------:|
| YOLOv3-Tiny |  46.98% |   6.2   |
| YOLOv4-Tiny |  58.03% |   12.3  |
|   YOLOv5s   |   68%   |   5.6   |


### Download the trained model
##### YOLOv5s(640)
[huawei-trash-yolov5s.pt](https://drive.google.com/file/d/1h4deB3p72AKJO8mgVJzCLNMx0NtIng0X/view?usp=sharing)

##### YOLOv5s(416)
[huawei-trash-yolov5s-416.pt](https://drive.google.com/file/d/1QDDnsZXbxXQNh4qsL_vda9WNA7bRkrQJ/view?usp=sharing)

##### YOLOv4-tiny
[yolov4-tiny-huawei_best.weights](https://drive.google.com/file/d/1_FXjjZ90qajkZPdndGuMHgHU1Yl0Sn33/view?usp=sharing)

##### YOLOv3-tiny
[yolov3-tiny-huawei_best.weights](https://drive.google.com/file/d/1ErTMlDM4TcJadcbNsQEhO3w0_qjCp4AV/view?usp=sharing)
