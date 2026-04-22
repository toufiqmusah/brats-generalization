import os
import shutil

def collect_seg_files(src_root, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    for root, dirs, files in os.walk(src_root):
        for file in files:
            if file.endswith("-seg.nii.gz"):
                src_path = os.path.join(root, file)

                dst_filename = file
                dst_path = os.path.join(dst_root, dst_filename)
                shutil.move(src_path, dst_path)
                print(f"Copied: {src_path} → {dst_path}")

if __name__ == "__main__":
    source_folder = "/content/BraTS-Africa"
    destination_folder = "/content/BraTS-Africa/BraTS-Africa-Seg-GT"

    collect_seg_files(source_folder, destination_folder)