# balance_dataset.py
import os
import random
import shutil

def balance_dataset(input_dir, output_dir, max_files_per_class=400):
    os.makedirs(output_dir, exist_ok=True)

    for class_name in ['Violence', 'Non-Violence']:
        src = os.path.join(input_dir, class_name)
        dst = os.path.join(output_dir, class_name)
        os.makedirs(dst, exist_ok=True)

        all_files = os.listdir(src)
        selected_files = random.sample(all_files, min(max_files_per_class, len(all_files)))

        for file in selected_files:
            shutil.copy(os.path.join(src, file), os.path.join(dst, file))

        print(f"✅ Balanced {class_name}: {len(selected_files)} files copied to {dst}")

# Example usage:
# balance_dataset("dataset/train", "dataset/train_balanced", max_files_per_class=400)
