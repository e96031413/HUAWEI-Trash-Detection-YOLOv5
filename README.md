# HUAWEI-Trash-Detection-YOLOv5
Training dataset from Huawei Cloud competition 2020 with YOLOv5

### Download the original dataset from BaiduYun
The dataset source is from [this post](https://blog.csdn.net/qq_38410428/article/details/106974147)

link: https://pan.baidu.com/s/1lh1D1wvXUV3rjJOUpWsBlA 

password: znk3

### Download the post-process dataset from Google Drive(Data format conversion part)
You can download the processed dataset [here](https://drive.google.com/file/d/1E6JtyAt_TRSrK47d9F6-OtC9rrKFXsDc/view?usp=sharing)

Tha maximum image size is 5.25 MB, the minimum image size is 1.25 KB

**train:val:test = 80 : 10 : 10**

-train: 11971 images

-val: 1496 images

-test: 1497 images

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

#### with pre-trained weight
```
cd yolov5
python -m torch.distributed.launch --nproc_per_node 2 train.py --cfg models/yolov5s.yaml --img 640 --epochs 100 --batch-size 16 --data trash.yaml --weights 'yolov5s.pt' --devices 2,3
```
#### from scratch
```
cd yolov5
python -m torch.distributed.launch --nproc_per_node 2 train.py --cfg models/yolov5s.yaml --img 640 --epochs 100 --batch-size 16 --data trash.yaml --weights '' --devices 2,3
```
