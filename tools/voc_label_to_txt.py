import os
import numpy as np
import xmltodict

label_file = "/opt/datasets/cuhk/VOCdevkit/VOC2012/ImageSets/Main/val.txt"
anno_dir = "/opt/datasets/cuhk/VOCdevkit/VOC2012/Annotations"
target_file = "/opt/datasets/cuhk/VOCdevkit/VOC2012/val_labels.txt"

name_id_map = {"person": 0}

# name_id_map = {"aeroplane": 0,
#                "bicycle": 1,
#                "bird": 2,
#                "boat": 3,
#                "bottle": 4,
#                "bus": 5,
#                "car": 6,
#                "cat": 7,
#                "chair": 8,
#                "cow": 9,
#                "diningtable": 10,
#                "dog": 11,
#                "horse": 12,
#                "motorbike": 13,
#                "person": 14,
#                "pottedplant": 15,
#                "sheep": 16,
#                "sofa": 17,
#                "train": 18,
#                "tvmonitor": 19}

with open(label_file) as f:
    label_content = f.read()

label_content = label_content.strip("\n")
labels = label_content.split("\n")


def read_single_anno(anno_path):
    with open(anno_path) as f:
        xml_content = f.read()

    data = xmltodict.parse(xml_content)
    height = float(data["annotation"]["size"]["height"])
    width = float(data["annotation"]["size"]["width"])
    obj = data["annotation"]["object"]
    objects = [obj] if isinstance(obj, dict) else obj
    xmaxs = [float(item["bndbox"]["xmax"]) for item in objects]
    xmins = [float(item["bndbox"]["xmin"]) for item in objects]
    ymaxs = [float(item["bndbox"]["ymax"]) for item in objects]
    ymins = [float(item["bndbox"]["ymin"]) for item in objects]

    # xmaxs = xmaxs / width
    # xmins = xmins / width
    # ymaxs = ymaxs / height
    # ymins = ymins / height

    names = np.array([item["name"] for item in objects])
    ids = [name_id_map[item] for item in names]
    boxes = list(zip(ymins, xmins, ymaxs, xmaxs))
    boxes = np.array(boxes)
    boxes[..., [0, 2]] /= height
    boxes[..., [1, 3]] /= width
    boxes.tolist()
    return boxes, ids


lines = []

for item in labels:
    anno_path = os.path.join(anno_dir, item + ".xml")
    boxes, ids = read_single_anno(anno_path)
    boxes = np.array(boxes).astype(np.str).tolist()
    ids = np.array(ids).astype(np.str).tolist()

    boxes_ids = [",".join(box) + f",{id}" for box, id in zip(boxes, ids)]
    boxes_ids = " ".join(boxes_ids)
    line = f"{item} {boxes_ids}\n"
    print(line)
    lines.append(line)

with open(target_file, "w") as f:
    f.writelines(lines)
