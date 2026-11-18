# builds the 3 class subset used for training from the full UrbanSound8K download
# usage: python prepare_data.py --src UrbanSound8K --out UrbanSound_Filtered

import argparse
import os
import shutil

import pandas as pd

CLASSES = ["dog_bark", "drilling", "street_music"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="UrbanSound8K", help="unzipped UrbanSound8K folder (has audio/ and metadata/)")
    parser.add_argument("--out", default="UrbanSound_Filtered", help="where to write the filtered subset")
    args = parser.parse_args()

    meta = pd.read_csv(os.path.join(args.src, "metadata", "UrbanSound8K.csv"))
    meta = meta[meta["class"].isin(CLASSES)].reset_index(drop=True)

    # UrbanSound8K stores clips in audio/fold1 ... audio/fold10, copy them into one flat folder
    out_audio = os.path.join(args.out, "audio")
    os.makedirs(out_audio, exist_ok=True)
    for _, row in meta.iterrows():
        src = os.path.join(args.src, "audio", f"fold{row['fold']}", row["slice_file_name"])
        shutil.copy2(src, os.path.join(out_audio, row["slice_file_name"]))

    meta.to_csv(os.path.join(args.out, "filtered_metadata.csv"), index=False)
    print(f"copied {len(meta)} clips to {args.out}/")
    print(meta["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
