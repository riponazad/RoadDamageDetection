import os
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description ="Prepare the list of names of all train images for yolo.")

    parser.add_argument("--images_dir", default="", help="Specify the path to the train images dataset folder.")
    parser.add_argument("--out_name", default="", help="Specify the file name to save the list.")
    parser.add_argument("--annots_dir", default="", help="Specify the path to the annotaton (xmls) folder.")

    args = parser.parse_args()
    return args


def create_list(images_dir, annots_dir, out_name):
    img_files = list(sorted(os.listdir(images_dir)))
    annot_files = list(sorted(os.listdir(annots_dir)))

    n_samples = len(img_files)
    n_train = int(n_samples*0.75)

    with open(out_name+"_train.txt", "w") as f:
        for i in range(n_train):
            full_image_path = os.path.join(images_dir, img_files[i])
            f.write(str(full_image_path) + "\n")
    with open(out_name+"_val.txt", "w") as f:
        for i in range(n_train, n_samples):
            full_image_path = os.path.join(images_dir, img_files[i])
            f.write(str(full_image_path) + "\n")
    
    with open(out_name+"annot_train.txt", "w") as f:
        for i in range(n_train):
            full_annot_path = os.path.join(annots_dir, annot_files[i])
            f.write(str(full_annot_path) + "\n")
    with open(out_name+"annot_val.txt", "w") as f:
        for i in range(n_train, n_samples):
            full_annot_path = os.path.join(annots_dir, annot_files[i])
            f.write(str(full_annot_path) + "\n")


if __name__=='__main__':
    args = parse_args()

    create_list(args.images_dir, args.annots_dir, args.out_name)


