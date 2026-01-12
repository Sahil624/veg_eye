import torch
from ultralytics import YOLO
from pathlib import Path
import argparse
import yaml


DEFAULT_DATASET_YAML = "./data/dataset.yaml"
DEFAULT_IMG_SIZE = 640
DEFAULT_DEVICE = "auto"

# --- Training Parameters ---
DEFAULT_TRAINING_EPOCHS = 50
DEFAULT_TRAINING_BATCH_SIZE = 16
DEFAULT_PROJECT_NAME = "runs/train"
DEFAULT_EXPERIMENT_NAME = "exp"


def check_dataset_yaml(yaml_path):
    path_obj = Path(yaml_path)
    if not path_obj.is_file():
        print(f"\nERROR: Dataset YAML file not found at '{yaml_path}'")
        print(
            "Please provide the correct path using the --data argument or update DEFAULT_DATASET_YAML."
        )
        return False
    try:
        with open(yaml_path, "r") as f:
            data_cfg = yaml.safe_load(f)
            if not isinstance(data_cfg, dict):
                print(
                    f"ERROR: Dataset YAML file '{yaml_path}' is not a valid YAML dictionary."
                )
                return False
            # Basic checks for essential keys (adjust as needed for your YAML structure)
            # 'path' might be optional if train/val paths are absolute
            if "path" not in data_cfg and (
                not Path(data_cfg.get("train", ".")).is_absolute()
                or not Path(data_cfg.get("val", ".")).is_absolute()
            ):
                print(
                    f"Warning: 'path' key missing in '{yaml_path}' and train/val paths seem relative. Paths might be resolved incorrectly."
                )

            if not all(key in data_cfg for key in ["train", "val", "names"]):
                print(
                    f"ERROR: Dataset YAML '{yaml_path}' missing one or more required keys: 'train', 'val', 'names'."
                )
                return False
            if not isinstance(data_cfg["names"], (dict, list)):
                print(
                    f"ERROR: 'names' key in '{yaml_path}' should be a list or dictionary of class names."
                )
                return False

    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse dataset YAML file '{yaml_path}': {e}")
        return False
    except Exception as e:
        print(
            f"ERROR: An unexpected error occurred while checking dataset YAML '{yaml_path}': {e}"
        )
        return False
    return True


def train_model(
    model_name,
    dataset_yaml,
    epochs,
    batch_size,
    img_size,
    device,
    project_name,
    exp_name,
    optimizer='auto'
):
    """Trains a YOLO model."""
    print(f"\n--- Starting Training for: {model_name} ---")
    print(f"  Dataset: {dataset_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {device}")
    print(f"  Project: {project_name}")
    print(f"  Experiment Name: {exp_name}")

    try:
        model = YOLO(model_name)

        base_args = dict(
            data=dataset_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            device=device,
            project=project_name,
            name=exp_name,
            exist_ok=False,
            optimizer=optimizer
        )

        model.train(
            **base_args
        )
        print(f"\n--- Training Finished for: {model_name} ---")
        print(f"Results saved to: {Path(project_name) / exp_name}")

    except Exception as e:
        print(f"\nERROR during training for model {model_name}: {e}")
        import traceback

        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 Training and Baseline Measurement Script"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model to use. For 'train': Pretrained model name (e.g., 'yolov11n.pt') or path to start from. Required for train action.",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATASET_YAML,
        help=f"Path to the dataset configuration YAML file. Default: '{DEFAULT_DATASET_YAML}'",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help=f"Input image size (square). Default: {DEFAULT_IMG_SIZE}",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help=f"Device to run on, i.e. 'cpu' or '0' for GPU. Default: {DEFAULT_DEVICE}",
    )

    # Training specific arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_TRAINING_EPOCHS,
        help=f"Number of training epochs. Default: {DEFAULT_TRAINING_EPOCHS}",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_TRAINING_BATCH_SIZE,
        help=f"Training batch size (-1 for auto-batch). Default: {DEFAULT_TRAINING_BATCH_SIZE}",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=DEFAULT_PROJECT_NAME,
        help=f"Directory to save training runs. Default: '{DEFAULT_PROJECT_NAME}'",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"Specific name for the training experiment run. Default: '{DEFAULT_EXPERIMENT_NAME}'",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='auto',
        help=f"Optimizer for training. Default: 'auto'",
    )

    args = parser.parse_args()

    if not check_dataset_yaml(args.data):
        exit(1)

    # Determine actual device if 'auto'
    resolved_device = args.device
    if args.device == "auto":
        if torch.cuda.is_available():
            resolved_device = "cuda"
        else:
            resolved_device = "cpu"
    elif "," in args.device:
        resolved_device = list(map(int, args.device.split(",")))

    print(f"Effective Device: {resolved_device}")

    train_model(
        model_name=args.model,
        dataset_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=resolved_device,  # Use resolved device
        project_name=args.project,
        exp_name=args.name,
        optimizer=args.optimizer
    )


if __name__ == "__main__":
    main()
