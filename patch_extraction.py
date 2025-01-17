import os
import numpy as np
from glob import glob
from visualize_slide_and_annotations import extract_tissue_patches, analyze_patches_with_instanseg
from tqdm import tqdm
from instanseg import InstanSeg
import matplotlib.pyplot as plt


patch_directory = "moffit_data/patches"
manchenko_normalize = False
save_nuclei_plots = True
plots_directory = os.path.join(patch_directory, "nuclei_plots")
os.makedirs(patch_directory, exist_ok=True)
if save_nuclei_plots:
    os.makedirs(plots_directory, exist_ok=True)
WSIs = glob(os.path.join("moffit_data", "*", "*.svs"))

for idx, WSI in tqdm(enumerate(WSIs), total=len(WSIs), desc="Extracting patches"):
    xml_path = WSI.replace(".svs", ".xml")
    if manchenko_normalize:
        raise NotImplementedError("Manchenko normalization not implemented")
        if idx == 0:
            patches, coords = extract_tissue_patches(
                WSI,
                xml_path,
                patch_size=256,
                level=0, 
                normalize=True,
                reference_patch=reference)
            reference = patches[0]  # Use first patch as reference
        else:
            patches, coords = extract_tissue_patches(
                WSI,
                xml_path,
                patch_size=256,
                level=0,
                normalize=True,
                reference_patch=reference)
    else:
        patches, coords = extract_tissue_patches(
            WSI,
            xml_path,
            patch_size=256,
            level=0,
        )
    nuclei_stats, labeled_outputs = analyze_patches_with_instanseg(patches, WSI)

    for pidx, patch in enumerate(patches):
        patch_path = os.path.join(patch_directory, f"{idx}_{pidx}.npy")
        np.save(patch_path, patch)
        
        # Save nuclei visualization plots if enabled
        if save_nuclei_plots and 'nuclei_contours' in coords[pidx]:
            f = plt.figure(figsize=(8, 8))
            plt.imshow(patch)
            # Plot contours for each nucleus
            for contour in coords[pidx]['nuclei_contours']:
                plt.plot(contour[:, 0], contour[:, 1], 'r-', linewidth=1)
            plt.axis('off')
            plot_path = os.path.join(plots_directory, f"{idx}_{pidx}_nuclei.png")
            plt.show()
            os._exit()
            plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
            plt.close(f)
