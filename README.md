### Using Anaconda to estublish a visual environment.

       

```python
pip install ultralytics
pip install fastdeploy-gpu-python -f https://link.zhihu.com/?target=https%3A//www.paddlepaddle.org.cn/whl/fastdeploy.html

```

1. Download the 
2. (Prepare your own data set);准备自己的数据集:

       (Dataset format requirements);数据集格式需求：

```
dataset #((Dataset name: e.g. fire)); (数据集名字：例如fire)
├── images
       ├── train
              ├── xx.jpg
       ├── val
              ├── xx.jpg
├── labels
       ├── train
              ├── xx.txt
       ├── val
              ├── xx.txt
```

(Data set partitioning); 数据集划分：

```python
# coding:utf-8

import os
import random
import argparse

parser = argparse.ArgumentParser()
#(The address of the xml file, modify it according to your own data. xml is generally stored under Annotations.); xml文件的地址，根据自己的数据进行修改 xml一般存放在Annotations下
parser.add_argument('--xml_path', default='Annotations', type=str, help='input xml label path')
# (To divide the data set, select ImageSets/Main under your own data as the address.); 数据集的划分，地址选择自己数据下的ImageSets/Main
parser.add_argument('--txt_path', default='ImageSets/Main', type=str, help='output txt label path')
opt = parser.parse_args()

trainval_percent = 1.0
train_percent = 0.9
xmlfilepath = opt.xml_path
txtsavepath = opt.txt_path
total_xml = os.listdir(xmlfilepath)
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)

num = len(total_xml)
list_index = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list_index, tv)
train = random.sample(trainval, tr)

file_trainval = open(txtsavepath + '/trainval.txt', 'w')
file_test = open(txtsavepath + '/test.txt', 'w')
file_train = open(txtsavepath + '/train.txt', 'w')
file_val = open(txtsavepath + '/val.txt', 'w')

for i in list_index:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        file_trainval.write(name)
        if i in train:
            file_train.write(name)
        else:
            file_val.write(name)
    else:
        file_test.write(name)

file_trainval.close()
file_train.close()
file_val.close()
file_test.close()
```

.xml to .txt

```python
# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd

sets = ['train', 'val', 'test']
classes = ["a", "b"]   # (Change to your own category); 改成自己的类别
abs_path = os.getcwd()
print(abs_path)

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(image_id):
    in_file = open('/home/trainingai/zyang/yolov5/paper_data/Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open('/home/trainingai/zyang/yolov5/paper_data/labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('Difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # (Mark out of bounds correction); 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()
for image_set in sets:
    if not os.path.exists('/home/trainingai/zyang/yolov5/paper_data/labels/'):
        os.makedirs('/home/trainingai/zyang/yolov5/paper_data/labels/')
    image_ids = open('/home/trainingai/zyang/yolov5/paper_data/ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open('/home/trainingai/zyang/yolov5/paper_data/%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        list_file.write(abs_path + '/paper_data/images/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
```

### **Add data configuration file**
### **添加数据配置文件**

Create a new myvoc.yaml in the yolov5/data folder
在yolov5/data文件夹下新建myvoc.yaml

The content is as follows：

```python
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: E:\for_test_proj\yolov5_ncnn\datasets\fire  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')
test:  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['fire']  # class names
```

(Attention please, the train and val parameters here are the paths of the images themselves)
(注意，这里的train,val参数是图片本身的路径）

Tips: ./model/yolov5s.yaml, (Just change the category and change it to your own number of categories.); 中改一下类别就可以了，改成你自己的类别数量。

![e3205378ea0144a5a8d8f8b735612d02.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/13ddcf97-d85b-488c-b6ea-cd24827b69e0/e3205378ea0144a5a8d8f8b735612d02.png)

### Edit the model’s configuration file:
### 编辑模型的配置文件：

***Clustering to obtain a priori boxes (optional) (clustering to regenerate anchors takes a long time) *The latest version of yolov5, it will automatically calculate anchors using kmeans**
***聚类得出先验框（可选）（聚类重新生成anchors运行时间较长）*最新版的yolov5，它会自动kmeans算出anchors**

The manual calculation process is as follows:
手动计算过程如下：

Create a new kmeans.py under the dataset folder
在数据集文件夹下新建kmeans.py

```python
import numpy as np

def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")                 # (If this error is reported, you can change this line to pass.); 如果报这个错，可以把这行改成pass即可

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)

    return iou_

def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)

def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)

        nearest_clusters = np.argmin(distances, axis=1)

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

        last_clusters = nearest_clusters

    return clusters

if __name__ == '__main__':
    a = np.array([[1, 2, 3, 4], [5, 7, 6, 8]])
    print(translate_boxes(a))
```

Clustering generates the file clauculate_anchors.py of new anchors. The code content is as follows:
聚类生成新anchors的文件clauculate_anchors.py，代码内容如下：

```python
# -*- coding: utf-8 -*-
# Find the a priori frame based on the label file
# 根据标签文件求先验框

import os
import numpy as np
import xml.etree.cElementTree as et
from kmeans import kmeans, avg_iou

FILE_ROOT = "/home/trainingai/zyang/yolov5/paper_data/"     # (root path); 根路径
ANNOTATION_ROOT = "Annotations"  # (Dataset label folder path); 数据集标签文件夹路径
ANNOTATION_PATH = FILE_ROOT + ANNOTATION_ROOT

ANCHORS_TXT_PATH = "/home/trainingai/zyang/yolov5/data/anchors.txt"

CLUSTERS = 9
CLASS_NAMES = ['a', 'b']

def load_data(anno_dir, class_names):
    xml_names = os.listdir(anno_dir)
    boxes = []
    for xml_name in xml_names:
        xml_pth = os.path.join(anno_dir, xml_name)
        tree = et.parse(xml_pth)

        width = float(tree.findtext("./size/width"))
        height = float(tree.findtext("./size/height"))

        for obj in tree.findall("./object"):
            cls_name = obj.findtext("name")
            if cls_name in class_names:
                xmin = float(obj.findtext("bndbox/xmin")) / width
                ymin = float(obj.findtext("bndbox/ymin")) / height
                xmax = float(obj.findtext("bndbox/xmax")) / width
                ymax = float(obj.findtext("bndbox/ymax")) / height

                box = [xmax - xmin, ymax - ymin]
                boxes.append(box)
            else:
                continue
    return np.array(boxes)

if __name__ == '__main__':

    anchors_txt = open(ANCHORS_TXT_PATH, "w")

    train_boxes = load_data(ANNOTATION_PATH, CLASS_NAMES)
    count = 1
    best_accuracy = 0
    best_anchors = []
    best_ratios = []

    for i in range(10):      ##### (You can modify it, don’t make it too big, otherwise it will take a long time); 可以修改，不要太大，否则时间很长
        anchors_tmp = []
        clusters = kmeans(train_boxes, k=CLUSTERS)
        idx = clusters[:, 0].argsort()
        clusters = clusters[idx]
        # print(clusters)

        for j in range(CLUSTERS):
            anchor = [round(clusters[j][0] * 640, 2), round(clusters[j][1] * 640, 2)]
            anchors_tmp.append(anchor)
            print(f"Anchors:{anchor}")

        temp_accuracy = avg_iou(train_boxes, clusters) * 100
        print("Train_Accuracy:{:.2f}%".format(temp_accuracy))

        ratios = np.around(clusters[:, 0] / clusters[:, 1], decimals=2).tolist()
        ratios.sort()
        print("Ratios:{}".format(ratios))
        print(20 * "*" + " {} ".format(count) + 20 * "*")

        count += 1

        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_anchors = anchors_tmp
            best_ratios = ratios

    anchors_txt.write("Best Accuracy = " + str(round(best_accuracy, 2)) + '%' + "\r\n")
    anchors_txt.write("Best Anchors = " + str(best_anchors) + "\r\n")
    anchors_txt.write("Best Ratios = " + str(best_ratios))
    anchors_txt.close()
```

After running clauculate_anchors.py, a file anchors.txt will be generated, which contains the suggested a priori frame anchors.
运行clauculate_anchors.py跑完会生成一个文件 anchors.txt，里面有得出的建议先验框anchors. 

### Modify the path and parameters in train.py
### 修改train.py中的路径及参数

The parameters are explained as follows:
Epochs: refers to how many times the entire data set will be iterated during the training process. If the graphics card is not good, you can adjust it smaller.
Batch-size: How many pictures are viewed at one time before the weights are updated. Gradient descent mini-batch. If the graphics card is not good, you can adjust it smaller.
cfg: Configuration file that stores the model structure
data: files that store training and test data
img-size: Enter the image width and height. If the graphics card is not good, you can adjust it smaller.
rect: perform rectangular training
resume: Resume the recently saved model to start training
nosave: only save the final checkpoint
notest: only test the last epoch
evolve: evolve hyperparameters
bucket: gsutil bucket
cache-images: cache images to speed up training
weights: weight file path
name: Rename results.txt to results_name.txt
device: cuda device, i.e. 0 or 0,1,2,3 or cpu
adam: Use adam optimization
multi-scale: multi-scale training, img-size +/- 50%
single-cls: single-category training set

参数解释如下：
epochs：指的就是训练过程中整个数据集将被迭代多少次,显卡不行你就调小点。
batch-size：一次看完多少张图片才进行权重更新，梯度下降的mini-batch,显卡不行你就调小点。
cfg：存储模型结构的配置文件
data：存储训练、测试数据的文件
img-size：输入图片宽高,显卡不行你就调小点。
rect：进行矩形训练
resume：恢复最近保存的模型开始训练
nosave：仅保存最终checkpoint
notest：仅测试最后的epoch
evolve：进化超参数
bucket：gsutil bucket
cache-images：缓存图像以加快训练速度
weights：权重文件路径
name： 重命名results.txt to results_name.txt
device：cuda device, i.e. 0 or 0,1,2,3 or cpu
adam：使用adam优化
multi-scale：多尺度训练，img-size +/- 50%
single-cls：单类别的训练集

### Reference:

1. https://blog.csdn.net/qq_36756866/article/details/109111065
2. [https://blog.csdn.net/weixin_70251903/article/details/126813751?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-126813751-blog-122874005.235^v38^pc_relevant_sort_base1&spm=1001.2101.3001.4242.3&utm_relevant_index=7](https://blog.csdn.net/weixin_70251903/article/details/126813751?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-4-126813751-blog-122874005.235%5Ev38%5Epc_relevant_sort_base1&spm=1001.2101.3001.4242.3&utm_relevant_index=7)
3. [https://blog.csdn.net/qq_52859223/article/details/123701798?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-123701798-blog-109111065.235^v38^pc_relevant_sort_base1&spm=1001.2101.3001.4242.1&utm_relevant_index=1](https://blog.csdn.net/qq_52859223/article/details/123701798?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-123701798-blog-109111065.235%5Ev38%5Epc_relevant_sort_base1&spm=1001.2101.3001.4242.1&utm_relevant_index=1)
4. https://zhuanlan.zhihu.com/p/501798155
