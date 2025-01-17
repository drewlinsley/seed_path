import os
from pathlib import Path
import numpy as np
from visualize_slide_and_annotations import train_tumor_classifier
import pickle


def main():
    # Define paths
    data_dir = Path("moffit_data")
    slides = list((data_dir / "Failed_Slides_DNN").glob("*.svs"))
    slides += list((data_dir / "Pass_Slides_DNN").glob("*.svs"))

    # Winnow down to those slides with tumor annotations
    tumor_annotations = list((data_dir / "Failed_tumor_annotations").glob("*.xml"))
    tumor_annotations += list((data_dir / "Pass_tumor_annotations").glob("*.xml"))
    tumor_names = [str(x).split(os.path.sep)[-1].split(".")[0] for x in tumor_annotations]

    usable_slides, usable_tissue_annotations = [], []
    for s in slides:
        slide_name = str(s).split(os.path.sep)[-1].split(".")[0]
        if slide_name in tumor_names:
            usable_slides.append(s)
            usable_tissue_annotations.append(str(s).replace(".svs", ".xml"))    
    output_dir = Path("trained_models")
    output_dir.mkdir(exist_ok=True)
    
    # Get all SVS files from Pass_Slides_DNN
    import pdb;pdb.set_trace()
    classifier, accuracy = train_tumor_classifier(
        slide_dnn_paths=usable_slides,
        tissue_annotation_paths=usable_tissue_annotations,
        tumor_annotation_paths=tumor_annotations,
        patch_size=224,
        n_samples=1000  # Adjust this based on your needs
    )

            
    # Save the classifier
    model_path = output_dir / "classifier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    # Print summary
    print("Accuracy: ", accuracy)


if __name__ == "__main__":
    main() 