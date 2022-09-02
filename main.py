import os
import shutil

from yolov5.detect import run
from yolov5.model import load_model
import json

model = load_model(weights=r"C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Models\yolov5x.pt")


def test_annotate(path_labelme):
    from os import walk

    for dirpath, dirnames, filenames in walk(path_labelme):

        for filename in filenames:
            f = open(dirpath + r'\\' + filename)
            data = json.load(f)

            for i in range(len(data['shapes']) - 1, -1, -1):
                if data['shapes'][i]['label'] == 'Helmet':
                    data['shapes'].pop(i)

            # for shape in data['shapes']:
            #     if shape['label'] == "Helmet":
            #         idx = data['shapes'].index(shape)
            #         print(data['shapes'][idx])
            #         del data['shapes'][idx]

            with open(dirpath + r'\\' + filename, 'w') as outfile:
                json.dump(data, outfile)


def change_name():
    from os import walk, rename
    from os.path import exists

    path_images = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\Banco2'
    path_labelme = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\Banco2Labelme'

    count = 0

    for dirpath, dirnames, filenames in walk(path_labelme):
        for filename in filenames:
            name_split = filename.split('.')
            name_split[-1] = "jpg"
            name = '.'.join(name_split)

            if exists(path_images + r"\\" + name):
                rename(path_images + r"\\" + name, path_images + r"\\image_" + str(count) + r".jpg")
                rename(path_labelme + r"\\" + filename, path_labelme + r"\\image_" + str(count) + r".json")
                count += 1


# region delete people annotations
def xml_reader():
    path_annotations = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Annotation'
    path_images = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\JPEGImage'

    from os import walk, path
    import xmltodict
    from dicttoxml import dicttoxml

    for dirpath, dirnames, filenames in walk(path_annotations):
        count = 0
        for filename in filenames:
            with open(dirpath + r'\\' + filename, 'r', encoding='utf-8') as f:
                data = f.read()

            xml_dict_annotations = xmltodict.parse(data)
            name_image = filename.split('.')
            name_image = name_image[0] + ".jpg"
            flag = True
            count2 = 0

            if type(xml_dict_annotations['annotation']['object']) == list:

                for name in xml_dict_annotations['annotation']['object']:

                    if name['name'] == "helmet":
                        count2 += 1

                    if not name['name'] == "helmet":
                        flag = False

                        # if path.exists(path_images + r'\\' + name_image):
                        #     os.remove(path_images + r'\\' + name_image)
                        #
                        # if path.exists(dirpath + r'\\' + filename):
                        #     os.remove(dirpath + r'\\' + filename)
                        #
                        # print(name_image)

                if flag:
                    count += count2
                    shutil.move(path_images + r'\\' + name_image,
                                r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_images')

                    shutil.move(path_annotations + r'\\' + filename,
                                r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_annotations')

            elif type(xml_dict_annotations['annotation']['object']) == dict:

                if xml_dict_annotations['annotation']['object']['name'] == "helmet":
                    count2 += 1

                if not xml_dict_annotations['annotation']['object']['name'] == "helmet":
                    flag = False

                    # if path.exists(dirpath + r'\\' + filename):
                    #     os.remove(path_images + r'\\' + name_image)
                    #
                    # if path.exists(path_images + r'\\' + name_image):
                    #     os.remove(dirpath + r'\\' + filename)
                    #
                    # print(name_image)

                if flag:
                    count += count2
                    shutil.move(path_images + r'\\' + name_image,
                                r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_images')

                    shutil.move(path_annotations + r'\\' + filename,
                                r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_annotations')
            if count > 1000:
                break

def name_images():
    from os import walk, rename

    path_images = r'C:\Users\estagio.sst17\source\repos\AutomaticAnnotateImages\Annotations'
    cont = 0

    for dirpath, dirnames, filenames in walk(path_images):
        for filename in filenames:
            new_name_split = filename.split('_')
            new_name_tmp = new_name_split[-1]
            new_name = '_' + str(cont) + '.' + str(new_name_tmp.split('.')[-1])
            rename(dirpath + '\\' + filename, dirpath + '\\' + ("image" + str(new_name)))
            cont += 1

# endregion

# region xml to labelme
import os
import argparse

path_images = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_images\\'
save_dir = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_Labelme\\'


def read_xml_gtbox_and_label(xml_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)
    points = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        pose = obj.find('pose').text
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        box = [xmin, ymin, xmax, ymax]
        point = [cls, box]
        points.append(point)
    return points, width, height


def parse_args():
    parser = argparse.ArgumentParser(description='xml2json')
    parser.add_argument('--raw_label_dir', help='the path of raw label', default='')
    parser.add_argument('--pic_dir', help='the path of picture', default='')
    parser.add_argument('--save_dir', help='the path of new label', default='')
    args = parser.parse_args()
    return args


def df2labelme():
    import glob
    import json
    from os import walk
    from tqdm import tqdm

    path_annotations = r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_annotations\\'

    for dirpath, dirnames, labels in walk(path_annotations):
        for i, label_abs in tqdm(enumerate(labels), total=len(labels)):
            _, label = os.path.split(label_abs)
            label_name = label.rstrip('.xml')
            # img_path = os.path.join(args.pic_dir, label_name + '.jpg')
            #img_path = path_images + label_name + '.jpg'
            points, width, height = read_xml_gtbox_and_label(dirpath + label_abs)
            json_str = {}
            json_str['version'] = '4.5.6'
            json_str['flags'] = {}
            shapes = []
            for i in range(len(points)):
                # Determine whether the point in the lower left corner is the key point
                if points[i][0] == "left head":
                    shape = {}
                    shape['label'] = 'head'
                    shape['points'] = [[points[i][1][0], points[i][1][3]]]
                    shape['group_id'] = None
                    # Point type
                    shape['shape_type'] = 'point'
                    shape['flags'] = {}
                    shapes.append(shape)
                # The point at the lower right corner is the key point
                elif points[i][0] == "right head":
                    shape = {}
                    shape['label'] = 'head'
                    shape['points'] = [[points[i][1][2], points[i][1][3]]]
                    shape['group_id'] = None
                    shape['shape_type'] = 'point'
                    shape['flags'] = {}
                    shapes.append(shape)
                # The rest
                else:
                    shape = {}
                    shape['label'] = points[i][0] if points[i][0] != "with_mask" else "mask"
                    shape['points'] = [[points[i][1][0], points[i][1][1]],
                                       [points[i][1][2], points[i][1][3]]]
                    shape['group_id'] = None
                    # The dimension types of labelimg are basically rectangular
                    shape['shape_type'] = 'rectangle'
                    shape['flags'] = {}
                    shapes.append(shape)
            json_str['shapes'] = shapes
            json_str['imagePath'] = label_name + '.jpg'
            json_str['imageData'] = None
            json_str['imageHeight'] = height
            json_str['imageWidth'] = width
            with open(os.path.join(save_dir, label_name + '.json'), 'w') as f:
                json.dump(json_str, f, indent=2)


# endregion


def call_annotate():
    return run(
        source=r'C:\Users\estagio.sst17\OneDrive - SESIMS\Documentos\EPIs\Dataset_HardHat_Havard\Hardhat\Test\Helmet_images',
        model=model,
        save_txt=True,
        conf_thres=0.70,
        classes=[0])


if __name__ == '__main__':
    call_annotate()
