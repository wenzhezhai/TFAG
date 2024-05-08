import os

def space2line(img_dirs="input_imgs"):
    for root, _, files in os.walk(img_dirs):
        for filename in files:
            if  filename.endswith("png"):
                    if "mask" not in filename:
                        if " " in filename or "._" in filename or ".." in filename:
                            src_file = os.path.join(root, filename)
                            dest_file = os.path.join(root, filename.replace(" ", "_")).replace("._", "__").replace("..", ".")
                            print(src_file + " ---> " + dest_file)
                            os.rename(src_file, dest_file)
                        else:
                            print(filename + "is ok.")
                    

if __name__ == "__main__":

    space2line(img_dirs="input_imgs/mini_test_11")
    