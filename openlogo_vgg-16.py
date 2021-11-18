# 导入所需包
from __future__ import division, print_function, absolute_import
import numpy as np
import os
from PIL import Image
from collections import defaultdict
from itertools import product
from sklearn.model_selection import train_test_split
import shutil
import re
import glob
from scipy import ndimage
import pickle
from six.moves import cPickle as pickle
from six.moves import range
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
# 参数
posshiftshift_min = -5
posshiftshift_max = 5
scales = [0.9, 1.1]
rot_min = -15
rot_max = 15

# 导入数据
dir = 'D:/Third Semester/data/flickr_logos_27_dataset/'
imgdir = os.path.join(dir, 'flickr_logos_27_dataset_images')
pp_dir = os.path.join(dir, 'processedF')
annot = 'flickr_logos_27_dataset_training_set_annotation.txt'
annot_train = np.loadtxt(os.path.join(dir, annot), dtype='a')
print('train_annotation: %d, %d ' % (annot_train.shape))

def parse_annot(annot):
    fn = annot[0].decode('utf-8')
    class_name = annot[1].decode('utf-8')
    train_subset_class = annot[2].decode('utf-8')
    return fn, class_name, train_subset_class

# 提取TXT文件中给出的标志的x和y坐标
def get_rect(annot):
    rect = defaultdict(int)
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cx, cy, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    rect['x1'] = x1
    rect['y1'] = y1
    rect['x2'] = x2
    rect['y2'] = y2
    rect['cx'] = cx
    rect['cy'] = cy
    rect['wid'] = wid
    rect['hgt'] = hgt
    return rect

# 变换提取的logo的位置
def posshift(annot, im):
    posshift_ims = []
    posshift_suffixes = []

    rect = get_rect(annot)
    for sx, sy in product(
            range(posshiftshift_min, posshiftshift_max),
            range(posshiftshift_min, posshiftshift_max)):
        cx = rect['cx'] + sx
        cy = rect['cy'] + sy
        cropped_im = im.crop((cx - rect['wid'] // 2, cy - rect['hgt'] // 2,
                              cx + rect['wid'] // 2, cy + rect['hgt'] // 2))
        resized_im = cropped_im.resize((width, height))
        posshift_ims.append(resized_im)
        posshift_suffixes.append('p' + str(sx) + str(sy))
        cropped_im.close()

    return posshift_ims, posshift_suffixes

# 用于调整提取的logo的大小
def scale(annot, im):
    scale_ims = []
    scale_suffixes = []

    rect = get_rect(annot)
    for s in scales:
        w = int(rect['wid'] * s)
        h = int(rect['hgt'] * s)
        cropped_im = im.crop((rect['cx'] - w // 2, rect['cy'] - h // 2,
                              rect['cx'] + w // 2, rect['cy'] + h // 2))
        resized_im = cropped_im.resize((width, height))
        scale_ims.append(resized_im)
        scale_suffixes.append('s' + str(s))
        cropped_im.close()

    return scale_ims, scale_suffixes

# 旋转提取的logo
def rotate(annot, im):
    rotate_ims = []
    rotate_suffixes = []

    rect = get_rect(annot)
    for r in range(rot_min, rot_max):
        rotated_im = im.rotate(r)
        cropped_im = rotated_im.crop(
            (rect['cx'] - rect['wid'] // 2, rect['cy'] - rect['hgt'] // 2,
             rect['cx'] + rect['wid'] // 2, rect['cy'] + rect['hgt'] // 2))
        resized_im = cropped_im.resize((width, height))
        rotate_ims.append(resized_im)
        rotate_suffixes.append('r' + str(r))
        rotated_im.close()
        cropped_im.close()

    return rotate_ims, rotate_suffixes

# 用于剪裁logo
def crop(annot, im):
    x1, y1, x2, y2 = rect_coord(annot[3:])
    cropped_im = im.crop((x1, y1, x2, y2))
    cropped_im = cropped_im.resize((width, height))
    cropped_suffix = 'p00'
    return [cropped_im], [cropped_suffix]

# 应用于每个列表项并收集所有返回值
def rect_coord(annot_part):
    return list(map(int, annot_part))

def center_wid_hgt(x1, y1, x2, y2):
    cx = x1 + (x2 - x1) // 2
    cy = y1 + (y2 - y1) // 2
    wid = (x2 - x1)
    hgt = (y2 - y1)
    return cx, cy, wid, hgt

def is_skip(annot_part):
    x1, y1, x2, y2 = rect_coord(annot_part)
    _, _, wid, hgt = center_wid_hgt(x1, y1, x2, y2)
    if wid <= 0 or hgt <= 0:
        return True
    else:
        return False

# 保存处理后的图像
def save_im(annot, cnt, *args):
    fn, class_name, train_subset_class = parse_annot(annot)
    dst_dir = os.path.join(pp_dir, class_name)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for i, arg in enumerate(args):
        for im, suffix in zip(arg[0], arg[1]):
            save_fn = '_'.join([
                fn.split('.')[0], class_name, train_subset_class, str(cnt),
                suffix
            ]) + os.path.splitext(fn)[1]
            im.save(os.path.join(dst_dir, save_fn))


def close_im(*args):
    for ims in args:
        for im in ims:
            im.close()

# 调用所有子函数
def crop_and_aug(annot_train):
    cnt_per_file = defaultdict(int)
    for annot in annot_train:
        # 用于生成文件名
        fn, _, _ = parse_annot(annot)
        cnt_per_file[fn] += 1
        # 如果宽度或高度等于零则跳过
        if is_skip(annot[3:]):
            print('Skip: ', fn)
            continue
        # 打开图片
        im = Image.open(os.path.join(imgdir, fn))
        # 剪裁
        cropped_ims, cropped_suffixes = crop(annot, im)
        # 移动
        shifted_ims, shifted_suffixes = posshift(annot, im)
        # 缩放
        scaled_ims, scaled_suffixes = scale(annot, im)
        # 旋转
        rotated_ims, rotated_suffixes = rotate(annot, im)
        # 保存图片
        save_im(annot, cnt_per_file[fn], [cropped_ims, cropped_suffixes],
                [shifted_ims, shifted_suffixes], [scaled_ims, scaled_suffixes],
                [rotated_ims, rotated_suffixes])
        # 关闭图片
        close_im([im], cropped_ims, shifted_ims, scaled_ims, rotated_ims)

def crop_and_aug_with_none(annot_train, with_none=False):
    # 用根目录保存处理后的图像
    if not os.path.exists(pp_dir):
        os.makedirs(pp_dir)
    # 裁剪图像并应用增强
    crop_and_aug(annot_train)
    # 输出结果
    org_imgs = [img for img in os.listdir(imgdir)]
    crop_and_aug_imgs = [
        fname
        for root, dirs, files in os.walk(pp_dir)
        for fname in glob.glob(os.path.join(root, '*.jpg'))  # look for the file with .jpg extension.
    ]
    print('original: %d' % (len(org_imgs)))
    print('cropped: %d' % (len(crop_and_aug_imgs)))

# 用于将数据集拆分为2个目录Train和Test
def do_train_test_split():
    class_names = [cls for cls in os.listdir(pp_dir)]
    # 在特定的类名下创建目录
    for class_name in class_names:
        if os.path.exists(
                os.path.join(pp_dir, class_name, 'train')):
            continue
        if os.path.exists(
                os.path.join(pp_dir, class_name, 'test')):
            continue

        imgs = [
            img
            for img in os.listdir(
                os.path.join(pp_dir, class_name))
        ]
        # train=0.75, test=0.25
        train_imgs, test_imgs = train_test_split(imgs)
        # 移动图像到训练或测试目录
        # 创建目录
        os.makedirs(os.path.join(pp_dir, class_name, 'train'))
        os.makedirs(os.path.join(pp_dir, class_name, 'test'))
        for img in train_imgs:
            dst = os.path.join(pp_dir, class_name, 'train')
            src = os.path.join(pp_dir, class_name, img)
            # 移动图片到目录
            shutil.move(src, dst)
        for img in test_imgs:
            dst = os.path.join(pp_dir, class_name, 'test')
            src = os.path.join(pp_dir, class_name, img)
            shutil.move(src, dst)

# 裁剪和数据扩充
crop_and_aug_with_none(annot_train)
# 分割训练集和测试集
do_train_test_split()

# 图片参数
width = 32
height = 32
channel = 3
pix_val = 255.0

dir = 'D:/Third Semester/data/flickr_logos_27_dataset/'
# 存放处理过的图像的目录
pp_dir = os.path.join(dir, 'processedF')
# pickle文件的名称
pickle_file = 'logo_dataset.pickle'
# 处理后的图像作为训练集存储在pickle文件中的图像数量
train_size = 70000
val_size = 5000
# 处理后的图像作为测试集存储在pickle文件中的图像数量
test_size = 7000

# 创建数据集数组
def array(nb_rows, image_width, image_height, image_ch=1):
    if nb_rows:
        #  存储高度、宽度和通道
        dataset = np.ndarray((nb_rows, image_height, image_width, image_ch), dtype=np.float32)
        #  存储标签
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

# 将所有类的pickle文件合并到一个pickle文件中
def combine(pickle_files, train_size, val_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = array(val_size, width,
                                              height, channel)
    train_dataset, train_labels = array(train_size, width,
                                              height, channel)
    vsize_per_class = val_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                logo_set = pickle.load(f)
                np.random.shuffle(logo_set)
                if valid_dataset is not None:
                    valid_logo = logo_set[:vsize_per_class, :, :, :]
                    valid_dataset[start_v:end_v, :, :, :] = valid_logo
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class
                train_logo = logo_set[vsize_per_class:end_l, :, :, :]
                train_dataset[start_t:end_t, :, :, :] = train_logo
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise
    return valid_dataset, valid_labels, train_dataset, train_labels

def makepickle(train_dataset, train_labels, valid_dataset, valid_labels,
                test_dataset, test_labels):
    try:
        f = open(pickle_file, 'wb')
        save = {
            'train_dataset': train_dataset,
            'train_labels': train_labels,
            'valid_dataset': valid_dataset,
            'valid_labels': valid_labels,
            'test_dataset': test_dataset,
            'test_labels': test_labels,
        }
        # 保存图像的数据到一个pickle文件
        pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise

# 打开图片
def load_logo(data_dir):
    image_files = os.listdir(data_dir)
    dataset = np.ndarray(
        shape=(len(image_files), height, width, channel),
        dtype=np.float32)
    print(data_dir)
    num_images = 0
    for image in image_files:
        image_file = os.path.join(data_dir, image)
        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pix_val / 2) / pix_val
            if image_data.shape != (height, width, channel):
                raise Exception('Unexpected image shape: %s' %
                                str(image_data.shape))
            dataset[num_images, :, :] = image_data
            num_images = num_images + 1
        except IOError as e:
            print('Could not read:', image_file, ':', e,
                  '-it\'s ok, skipping.')

    dataset = dataset[0:num_images, :, :]
    print('Full dataset tensor:', dataset.shape)       # 特定类的处理图像数量
    print('Mean:', np.mean(dataset))                   # 计算整个类的平均值
    print('Standard deviation:', np.std(dataset))      # 计算整个类的标准差
    return dataset


# 创建pickle文件
def pickling(data_dirs, force=False):
    dataset_names = []
    for dir in data_dirs:
        set_filename = dir + '.pickle'
        dataset_names.append(set_filename)

        if os.path.exists(set_filename) and force:

            print('%s already present - Skipping pickling. ' % set_filename)
        else:
            print('Pickling %s.' % set_filename)
            dataset = load_logo(dir)
            try:
                with open(set_filename, 'wb') as f:
                    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', set_filename, ':', e)
    return dataset_names

def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels

CLASS_NAME = ['Apple', 'BMW', 'Heineken', 'HP', 'Intel', 'Mini', 'Starbucks', 'Vodafone', 'Citroen', 'Ferrari']

dirs = [os.path.join(pp_dir, class_name, 'train')   # 该类所有的训练折
        for class_name in CLASS_NAME]
test_dirs = [os.path.join(pp_dir, class_name, 'test')   # 该类所有的测试折
            for class_name in CLASS_NAME]

train_datasets = pickling(dirs)
test_datasets = pickling(test_dirs)
# 调用函数进行合并
valid_dataset, valid_labels, train_dataset, train_labels = combine(train_datasets, train_size, val_size)
test_dataset, test_labels = combine(test_datasets, test_size)
# 调用函数进行随机
train_dataset, train_labels = randomize(train_dataset, train_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
# 调用函数创建pickle文件
makepickle(train_dataset, train_labels, valid_dataset, valid_labels,test_dataset, test_labels)
statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

def read_data():
    with open("logo_dataset.pickle", 'rb') as f:
        save = pickle.load(f)
        X = save['train_dataset']       # 训练集
        Y = save['train_labels']        # 训练标签
        X_test = save['test_dataset']   # 测试集
        Y_test = save['test_labels']    # 测试标签
        del save
    return [X, X_test], [Y, Y_test]

def reformat(dataset, labels):
    # 重新格式化形状数组以为数据集提供标量值
    dataset = dataset.reshape((-1, 32, 32,3)).astype(np.float32)
    labels = (np.arange(10) == labels[:, None]).astype(np.float32)
    return dataset, labels

dataset, labels = read_data()
X,Y = reformat(dataset[0], labels[0])
X_test, Y_test = reformat(dataset[1], labels[1])
print('Training set', X.shape, Y.shape)
print('Test set', X_test.shape, Y_test.shape)
# 打乱数据
X, Y = shuffle(X, Y)
# 标准化
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()

# 定义CNN结构
# 32*32*3的许多图片作输入层
network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
# 卷积层1、2为64个滤波器，卷积核为3*3，默认填充为与原数组相同大小，激活函数选用relu
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
# 最大池化层1
network = max_pool_2d(network, 2)
# 随机失活层1，用于在训练过程中随机丢弃一些数据以防止过拟合，设定丢弃一半
network = dropout(network, 0.1)
# 卷积层3、4为128个滤波器，卷积核为3*3，默认填充为与原数组相同大小，激活函数选用relu
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
# 最大池化层2
network = max_pool_2d(network, 2)
# 随机失活层2，用于在训练过程中随机丢弃一些数据以防止过拟合，设定丢弃一半
network = dropout(network, 0.1)
# 卷积层5、6、7为256个滤波器，卷积核为3*3，默认填充为与原数组相同大小，激活函数选用relu
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
# 最大池化层3
network = max_pool_2d(network, 2)
# 随机失活层3，用于在训练过程中随机丢弃一些数据以防止过拟合，设定丢弃一半
network = dropout(network, 0.1)
# 卷积层8、9、10为512个滤波器，卷积核为3*3，默认填充为与原数组相同大小，激活函数选用relu
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
# 最大池化层4
network = max_pool_2d(network, 2)
# 随机失活层4，用于在训练过程中随机丢弃一些数据以防止过拟合，设定丢弃一半
network = dropout(network, 0.1)
# 卷积层11、12、13为512个滤波器，卷积核为3*3，默认填充为与原数组相同大小，激活函数选用relu
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
network = conv_2d(network, 512, 3, activation='relu')
# 最大池化层5
network = max_pool_2d(network, 2)
# 随机失活层5，用于在训练过程中随机丢弃一些数据以防止过拟合，设定丢弃一半
network = dropout(network, 0.1)
# 全连接层1为4096个结点，激活函数选用relu
network = fully_connected(network, 4096 , activation='relu')
# 全连接层2为4096个结点，激活函数选用relu
network = fully_connected(network, 4096 , activation='relu')
# 全连接层3为10个结点，激活函数选用sofmax，也即输出层
network = fully_connected(network, 10 , activation='softmax')

# 为了训练网络，我们将使用ADAM优化方法和分类交叉熵，在学习过程中确定损失并优化
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
# 在模型对象中覆盖网络
model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path="model\logo-classifier.tfl.ckpt")
# 10个epoch
model.fit(X,Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test), show_metric=True, batch_size=128, snapshot_epoch=True,
          run_id='logo-classifier')
# 当训练完成时将模型保存到一个文件中
model.save("logo-classifier.tfl")
print("Network trained and saved as logo-classifier.tfl!")
# 加载训练的数据集文件'logo-classifier.tfl.ckpt-5024'
model.load("model\logo-classifier.tfl.ckpt-5024")
# 评估模型
score=model.evaluate(X_test, Y_test)
print(score)
