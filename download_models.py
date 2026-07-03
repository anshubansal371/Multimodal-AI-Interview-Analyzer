import os
import gdown

MODEL_DIR = "models"

# Create folders
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(MODEL_DIR, "final_roberta_model"), exist_ok=True)

FILES = {
    os.path.join(MODEL_DIR, "face_model_best.keras"):
        "1sHM_dyOpWv2hqfBf0OZdns3OAzNKDjvq",

    os.path.join(MODEL_DIR, "audio_model_best.keras"):
        "1u6J5bV2C7Pg9ksxQcGcQgSgrRVWexTqC",

    os.path.join(MODEL_DIR, "fusion_model_best.keras"):
        "1rbyqBXoS80De3KiTaFCfEnd_1vV5eLp9",

    os.path.join(MODEL_DIR, "final_roberta_model", "model.safetensors"):
        "11q6HRwYldxKXnQuTVp1-oKeAnhBcaw3N",
    os.path.join(MODEL_DIR, "final_roberta_model", "config.json"):
        "1HbF_f1qmBJQmkCYkNYuTd7pm2la4qFqW",

    os.path.join(MODEL_DIR, "final_roberta_model", "tokenizer.json"):
        "1taavdT0HmTixuft8_2Rxl91zofRNMZ41",

    os.path.join(MODEL_DIR, "final_roberta_model", "tokenizer_config.json"):
        "1wNtoT9Ixy3OWhYskJz-G7zW7kPGjbF_l",

    os.path.join(MODEL_DIR, "final_roberta_model", "emotion_map.json"):
        "1g36env36ImhaFWi-la5gfWJjH81HAMRy",
}
def download_all():

    for path, file_id in FILES.items():

        if os.path.exists(path):
            print(f"✅ Already exists: {path}")
            continue

        os.makedirs(os.path.dirname(path), exist_ok=True)

        print(f"⬇ Downloading {os.path.basename(path)}")

        try:

            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                output=path,
                quiet=False,
            )

            if not os.path.exists(path):
                raise RuntimeError(
                    f"Failed to download {path}"
                )

            print(f"✅ Downloaded {path}")

        except Exception as e:

            print(f"❌ Error downloading {path}")
            print(e)

            raise


if __name__ == "__main__":
    download_all()
