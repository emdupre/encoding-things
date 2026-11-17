from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image
from tqdm.contrib.concurrent import thread_map
from transformers import AutoImageProcessor, AutoModel


def get_prediction(img_path):
    image = Image.open(img_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"]
    prediction = model(pixel_values)
    # torch.Size([1, 327684]), vertices in fsaverage,
    # first half is left hemisphere, second half right hemisphere
    # print(prediction.shape)

    return prediction.detach().numpy()


@click.command()
@click.option(
    "--sub_name",
    type=click.Choice(["subj01", "subj02", "subj05", "subj07"]),
    default="subj01",
    help="Subject name.",
)
@click.option(
    "--stimuli_dir",
    default="/home/emdupre/links/projects/def-pbellec/emdupre/NSD/stimuli",
    help="Stimulus directory.",
)
def main(subj_id, stimuli_dir):
    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    # whole brain model, 327684 vertices in fsaverage
    model = AutoModel.from_pretrained(
        "huzey/brain-synthesizer",
        trust_remote_code=True,
        subject=subj_id,  # options: subj01, subj02, subj05, subj07
    )

    stimuli_dir = Path(stimuli_dir)
    # "/Users/emdupre/Desktop/UdeM-Projects/things-encode/cneuromod.things.stimuli"
    # "/Users/emdupre/Desktop/UdeM-Projects/nsd/stimuli"
    output_dir = Path(stimuli_dir.parent, "stimuli.algonauts-predictions")
    output_dir.mkdir(exist_ok=True, parents=True)

    img_paths = Path(stimuli_dir).rglob("*.jpg")
    predictions = thread_map(get_prediction, img_paths, max_workers=10)

    # Save embeddings and file names to .npy files
    np.save(
        Path(
            output_dir,
            "brain-synthesizer-subj01-predictions.npy",
        ),
        np.array(predictions),
    )
    np.savetxt(
        Path(
            output_dir,
            "brain-synthesizer-subj01-file_names.txt",
        ),
        np.array([img.name for img in img_paths]),
        fmt="%s",
    )
    return


if __name__ == "__main__":
    main()
