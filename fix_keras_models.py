import zipfile
import json
import shutil
from pathlib import Path


ARTIFACTS_DIR = Path("recommended_artifacts")

MODEL_FILES = [
    "multimodal_model.keras",
    "coldstart_multimodal_model.keras",
]


def remove_quantization_config(obj):
    """
    Recursively remove quantization_config from Embedding
    layer configurations when it is None.
    """

    if isinstance(obj, dict):

        # Detect a Keras layer configuration
        if (
            obj.get("class_name") == "Embedding"
            and isinstance(obj.get("config"), dict)
        ):
            config = obj["config"]

            if config.get("quantization_config") is None:
                config.pop("quantization_config", None)

        # Continue recursively
        for value in obj.values():
            remove_quantization_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_quantization_config(item)


def patch_model(model_path: Path):

    backup_path = model_path.with_suffix(".keras.backup")

    print(f"\nProcessing: {model_path.name}")

    # Create backup
    shutil.copy2(model_path, backup_path)
    print(f"Backup created: {backup_path.name}")

    temp_path = model_path.with_suffix(".patched.keras")

    with zipfile.ZipFile(model_path, "r") as zin:

        # Read config.json
        config = json.loads(
            zin.read("config.json").decode("utf-8")
        )

        # Patch configuration
        remove_quantization_config(config)

        with zipfile.ZipFile(
            temp_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as zout:

            for item in zin.infolist():

                if item.filename == "config.json":

                    patched_config = json.dumps(
                        config,
                        ensure_ascii=False,
                        separators=(",", ":")
                    ).encode("utf-8")

                    zout.writestr(
                        item,
                        patched_config
                    )

                else:
                    zout.writestr(
                        item,
                        zin.read(item.filename)
                    )

    # Replace original
    shutil.move(temp_path, model_path)

    print(f"Patched successfully: {model_path.name}")


if __name__ == "__main__":

    if not ARTIFACTS_DIR.exists():
        raise FileNotFoundError(
            f"Artifacts directory not found: {ARTIFACTS_DIR}"
        )

    for model_file in MODEL_FILES:

        model_path = ARTIFACTS_DIR / model_file

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        patch_model(model_path)

    print("\nAll models patched.")