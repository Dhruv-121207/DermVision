from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

raw_dir = Path("data/raw")
metadata = pd.read_csv(raw_dir / "HAM10000_metadata.csv")

image_dir1 = raw_dir / "HAM10000_images_part_1"
image_dir2 = raw_dir / "HAM10000_images_part_2"

processed_dir = Path("data/processed")
train_dir = processed_dir / "train"
val_dir = processed_dir / "val"
test_dir = processed_dir / "test"

train_df, temp_df = train_test_split(
    metadata,
    test_size=0.30,
    stratify=metadata["dx"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["dx"],
    random_state=42
)

def copy_images(dataframe, output_dir):
    for _, row in dataframe.iterrows():
        class_name = row["dx"]
        image_name = row["image_id"] + ".jpg"

        class_folder = output_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)

        source = image_dir1 / image_name
        if not source.exists():
            source = image_dir2 / image_name

        destination = class_folder / image_name
        shutil.copy2(source, destination)

copy_images(train_df, train_dir)
copy_images(val_df, val_dir)
copy_images(test_df, test_dir)

print("Dataset prepared successfully!")
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")