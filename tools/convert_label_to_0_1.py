
import numpy as np

LABEL_FILE = "/opt/datasets/yymnist_yolo/yymnist_train.txt"
OUTPUT_FILE = "/opt/datasets/yymnist_yolo/train_labels.txt"

with open(LABEL_FILE) as f:
    content = f.read()

content = content.strip("\n")
lines = content.split("\n")

file_names = [line.split(" ")[0].split("/")[-1].split(".")[0] for line in lines]

array = [[item.split(",") for item in line.split(" ")[1:]] for line in lines]
array = [[[float(i) for i in item] for item in line] for line in array]

output_lines = []

for file_name, labels in zip(file_names, array):
    labels = np.array(labels)
    labels = np.concatenate((labels[..., 0:4] / 416, labels[:, 4:5]), -1)
    labels = labels[:, [1, 0, 3, 2, 4]]
    labels = labels[:, ]
    labels = labels.astype(np.str)
    labels = labels.tolist()
    labels = [",".join(item) for item in labels]
    labels = " ".join(labels)
    line = f"{file_name} {labels}\n"
    output_lines.append(line)
    print(file_name)


with open(OUTPUT_FILE, "w") as f:
    f.writelines(output_lines)

print("complete")


