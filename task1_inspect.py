import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

from waterbirds_model import CLASS_NAMES, load_model


EXAMPLES = {
    "Cape_May_Warbler_0048_162915.jpg": {
        "species": "Cape May Warbler",
        "true_class": "landbird",
        "environment": "forest / mossy branches",
    },
    "Common_Tern_0067_149540.jpg": {
        "species": "Common Tern",
        "true_class": "waterbird",
        "environment": "water",
    },
    "Slaty_Backed_Gull_0079_796020.jpg": {
        "species": "Slaty-backed Gull",
        "true_class": "waterbird",
        "environment": "sky and distant buildings",
    },
    "Sooty_Albatross_0036_796387.jpg": {
        "species": "Sooty Albatross",
        "true_class": "waterbird",
        "environment": "rocky / nest-like land",
    },
    "White_Pelican_0076_96427.jpg": {
        "species": "White Pelican",
        "true_class": "waterbird",
        "environment": "water",
    },
}


def find_assignment_images(zip_path):
    candidates = [Path(zip_path).expanduser()] if zip_path else [
        Path("Portfolio_2.zip"),
        Path("../Portfolio_2.zip"),
        Path.home() / "Downloads" / "Portfolio_2.zip",
    ]
    archive = next((path for path in candidates if path.is_file()), None)

    if archive is None:
        raise FileNotFoundError(
            f"Assignment archive not found: {zip_path or 'Portfolio_2.zip'}"
        )
    return archive


def extract_images(archive, temporary_directory):
    destination = Path(temporary_directory) / "Exercise_1" / "images"
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        for member in source.namelist():
            prefix = "Exercise_1/images/"
            if member.startswith(prefix) and not member.endswith("/"):
                target = destination / Path(member).name
                with source.open(member) as input_file, target.open("wb") as output_file:
                    output_file.write(input_file.read())
    return destination


def resolve_image_directory(image_directory, zip_path):
    if image_directory:
        directory = Path(image_directory).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {directory}")
        return directory

    local_directory = Path(__file__).resolve().parent / "images"
    if not zip_path and local_directory.is_dir():
        return local_directory

    return find_assignment_images(zip_path)


def predict(model, device, image_path):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0].cpu()

    predicted_index = int(probabilities.argmax().item())
    return image, predicted_index, probabilities


def save_montage(results, output_path):
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    for axis, result in zip(axes, results):
        axis.imshow(result["image"])
        status = "CORRECT" if result["correct"] else "INCORRECT"
        title = (
            f'{result["species"]}\n'
            f'True: {result["true_class"]} | '
            f'Predicted: {result["predicted_class"]}\n'
            f'{status}'
        )
        axis.set_title(title)
        axis.axis("off")

    for extra_axis in axes[len(results):]:
        extra_axis.axis("off")

    figure.suptitle("Task 1 - Customer examples and model predictions", fontsize=16)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Inspect customer images and model predictions")
    parser.add_argument("--images", type=str, help="Directory containing the five images")
    parser.add_argument("--zip", type=str, help="Path to the assignment ZIP")
    parser.add_argument(
        "--checkpoint",
        default=str(Path(__file__).resolve().parent / "waterbirds_erm_resnet18_biased.pt"),
    )
    parser.add_argument("--output-dir", type=str, default="task1_output")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    image_source = resolve_image_directory(args.images, args.zip)

    with tempfile.TemporaryDirectory() as temporary_directory:
        image_directory = (
            extract_images(image_source, temporary_directory)
            if zipfile.is_zipfile(image_source)
            else image_source
        )
        if not image_directory.is_dir():
            raise FileNotFoundError(f"No usable image directory: {image_directory}")

        rows = []
        results = []
        for filename, metadata in EXAMPLES.items():
            image_path = image_directory / filename
            image, predicted_index, probabilities = predict(model, device, image_path)
            predicted_class = CLASS_NAMES[predicted_index]
            row = {
                "filename": filename,
                "species": metadata["species"],
                "true_class": metadata["true_class"],
                "environment": metadata["environment"],
                "predicted_class": predicted_class,
                "landbird_probability": round(float(probabilities[0]), 6),
                "waterbird_probability": round(float(probabilities[1]), 6),
                "correct": predicted_class == metadata["true_class"],
            }
            rows.append(row)
            results.append({**row, "image": image})

        save_montage(results, output_dir / "task1_customer_examples.png")

        print(f"Device: {device}")
        print(f"Montage: {output_dir / 'task1_customer_examples.png'}")
        for row in rows:
            status = "CORRECT" if row["correct"] else "INCORRECT"
            print(
                f'{row["filename"]}: true={row["true_class"]}, '
                f'predicted={row["predicted_class"]}, '
                f'confidence=landbird:{row["landbird_probability"]:.6f}/'
                f'waterbird:{row["waterbird_probability"]:.6f} [{status}]'
            )


if __name__ == "__main__":
    sys.exit(main())
