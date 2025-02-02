import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import openslide
from pathlib import Path
from matplotlib.patches import Rectangle
from skimage import color
from matplotlib.path import Path as MatPath
from instanseg import InstanSeg
from tqdm import tqdm
from test_dino import vit_small
import torch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


instanseg_brightfield = InstanSeg("brightfield_nuclei", verbosity=0)
dino_model = vit_small(pretrained=True, progress=False, key="DINO_p16", patch_size=16)
mu = torch.from_numpy(np.asarray([ 0.70322989, 0.53606487, 0.66096631 ]))
std = torch.from_numpy(np.asarray([ 0.21716536, 0.26081574, 0.20723464 ]))


def read_xml_annotations(xml_path):
    """Read XML annotations and extract vertex coordinates."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get microns per pixel scale
    mpp = float(root.attrib.get('MicronsPerPixel', 1.0))
    
    # Extract all vertices from all regions
    regions = []
    for region in root.findall('.//Region'):
        vertices = []
        for vertex in region.findall('.//Vertex'):
            x = float(vertex.attrib['X'])
            y = float(vertex.attrib['Y'])
            vertices.append([x, y])
        if vertices:
            regions.append(np.array(vertices))
    
    return regions, mpp

def is_tissue(patch):
    """
    Determine if a patch contains tissue based on color and brightness.
    Returns True if the patch contains >= 90% tissue.
    """
    # Convert to HSV for better tissue detection
    hsv = color.rgb2hsv(patch)
    
    # Tissue is typically darker and more saturated than background
    # Adjust these thresholds based on your specific slides
    is_tissue_pixel = (hsv[:, :, 2] < 0.98) & (hsv[:, :, 1] > 0.02)
    tissue_percentage = np.mean(is_tissue_pixel)
    
    return tissue_percentage >= 0.9

def is_patch_in_annotation(x, y, patch_size, regions, scale_x, scale_y):
    """
    Check if a patch intersects with any annotation region.
    """
    patch_corners = np.array([
        [x, y],  # top-left
        [x + patch_size, y],  # top-right
        [x + patch_size, y + patch_size],  # bottom-right
        [x, y + patch_size],  # bottom-left
    ])
    
    for vertices in regions:
        # Scale vertices to match thumbnail coordinates
        scaled_vertices = vertices.copy()
        scaled_vertices[:, 0] *= scale_x
        scaled_vertices[:, 1] *= scale_y
        
        # Create path from vertices
        path = MatPath(scaled_vertices)
        
        # Check if any corner is inside the polygon
        if any(path.contains_points(patch_corners)):
            return True
        
        # Check if any polygon point is inside the patch
        patch_bounds = MatPath(patch_corners)
        if any(patch_bounds.contains_points(scaled_vertices)):
            return True
    
    return False

def plot_annotations_on_slide(svs_path, xml_path, patch_size=224, output_path=None, 
                            analyze_nuclei=False, debug_nuclei=False):
    """
    Plot annotations on slide with optional nuclei analysis.
    """
    # Open slide and get thumbnail for visualization
    slide = openslide.OpenSlide(str(svs_path))
    
    # Get a reasonable thumbnail size while maintaining aspect ratio
    thumb_size = 1000
    slide_dims = slide.dimensions
    aspect_ratio = slide_dims[1] / slide_dims[0]
    thumb_dims = (thumb_size, int(thumb_size * aspect_ratio))
    
    thumbnail = np.array(slide.get_thumbnail(thumb_dims))
    plt.figure(figsize=(10, 10))
    plt.imshow(thumbnail)
    
    # Calculate scale factors
    scale_x = thumb_dims[0] / slide_dims[0]
    scale_y = thumb_dims[1] / slide_dims[1]
    
    # Read and plot annotations
    regions, mpp = read_xml_annotations(xml_path)
    for vertices in regions:
        scaled_vertices = vertices.copy()
        scaled_vertices[:, 0] *= scale_x
        scaled_vertices[:, 1] *= scale_y
        plt.plot(scaled_vertices[:, 0], scaled_vertices[:, 1], 'r-', linewidth=2)
    
    # Extract patches that overlap with annotations
    patches, coordinates = extract_tissue_patches(svs_path, xml_path, patch_size=patch_size, level=0)
    
    # Reduce patches to tumor regions
    embs = []
    with torch.no_grad():
        for i, patch in enumerate(patches):
            x = torch.from_numpy(patch).float() / 255.
            z = ((x - mu[None, None]) / std[None, None]).float()
            emb = dino_model(z.permute(2, 0, 1)[None].float())
            embs.append(emb.cpu().numpy())
    embs = np.asarray(embs)
    pca = PCA(n_components=2)
    scores = pca.fit_transform(embs.squeeze())
    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(scores)
    labels = kmeans.labels_
    score_0 = scores[labels == 0, 0].max(0)
    score_1 = scores[labels == 1, 0].max(0)
    if score_0 > score_1:
        tumor_mask = labels == 0
    else:
        tumor_mask = labels == 1
    patches = np.asarray(patches)
    benign_patches = patches[~tumor_mask]
    benign_coordinates = np.asarray(coordinates)[~tumor_mask]
    patches = patches[tumor_mask]
    coordinates = np.asarray(coordinates)[tumor_mask]
    
    # Run InstanSeg analysis if requested
    nuclei_stats = None
    if analyze_nuclei and len(patches):
        nuclei_stats, labeled_outputs = analyze_patches_with_instanseg(patches, svs_path)
        
        # Plot tumor patches in green
        for (x, y), count in zip(coordinates, nuclei_stats['counts_per_patch']):
            thumb_x = x * scale_x
            thumb_y = y * scale_y
            thumb_size = patch_size * scale_x
            
            rect = Rectangle((thumb_x, thumb_y), thumb_size, thumb_size, 
                           fill=True, alpha=0.2, color='green')  # Tumor patches in green
            plt.gca().add_patch(rect)
            
            if debug_nuclei:
                plt.text(thumb_x + thumb_size/2, thumb_y + thumb_size/2, 
                        f'{count}', color='red', fontsize=7,
                        horizontalalignment='center',
                        verticalalignment='center')
        
        # Plot benign patches in blue
        for x, y in benign_coordinates:
            thumb_x = x * scale_x
            thumb_y = y * scale_y
            thumb_size = patch_size * scale_x
            
            rect = Rectangle((thumb_x, thumb_y), thumb_size, thumb_size, 
                           fill=True, alpha=0.2, color='blue')  # Benign patches in blue
            plt.gca().add_patch(rect)
    
    if output_path:
        plt.savefig(output_path)
    
    # Always show the plot if debug_nuclei is True, regardless of output_path
    if debug_nuclei:
        plt.show()
    else:
        plt.close()
    
    return nuclei_stats


def get_macenko_normalizer(reference_patch=None, target_stains=None):
    """
    Create a Macenko normalizer function. If no reference is provided, uses standard H&E values.
    
    Args:
        reference_patch: Optional reference image to define target stain vectors
        target_stains: Optional manually specified target stain vectors
    
    Returns:
        normalize_patch: Function that normalizes patches using Macenko's method
    """
    # Default H&E target values if none provided
    if target_stains is None:
        target_stains = np.array([
            [0.5626, 0.2159],  # Hematoxylin
            [0.7201, 0.8012],  # Eosin
        ])
    
    # If reference patch provided, compute its stain vectors
    if reference_patch is not None:
        # Convert to optical density
        od = -np.log((reference_patch.astype(float) + 1) / 256)
        
        # Remove pixels with very low optical density
        od_reshaped = od.reshape((-1, 3))
        od_thresh = od_reshaped[np.all(od_reshaped > 0.15, axis=1)]
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(od_thresh.T))
        eigvecs = eigvecs[:, [1, 2]]  # H&E eigenvectors
        
        # Project data onto eigenvectors
        proj = np.dot(od_thresh, eigvecs)
        
        # Find angle of each point
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        
        # Find extreme angles (min for H, max for E)
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        # Convert angles back to stain vectors
        target_stains = np.dot(np.array([
            [np.cos(min_phi), np.cos(max_phi)],
            [np.sin(min_phi), np.sin(max_phi)]
        ]).T, eigvecs.T)
    
    def normalize_patch(patch):
        """Normalize a single patch using Macenko's method."""
        # Convert to optical density
        od = -np.log((patch.astype(float) + 1) / 256)
        
        # Remove pixels with very low optical density
        od_reshaped = od.reshape((-1, 3))
        od_thresh = od_reshaped[np.all(od_reshaped > 0.15, axis=1)]
        
        if len(od_thresh) == 0:  # If no valid pixels, return original
            return patch
        
        # Compute eigenvectors
        _, eigvecs = np.linalg.eigh(np.cov(od_thresh.T))
        eigvecs = eigvecs[:, [1, 2]]  # H&E eigenvectors
        
        # Project data onto eigenvectors
        proj = np.dot(od_thresh, eigvecs)
        
        # Find angle of each point
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        
        # Find extreme angles
        min_phi = np.percentile(phi, 1)
        max_phi = np.percentile(phi, 99)
        
        # Convert angles back to stain vectors
        stain_vectors = np.dot(np.array([
            [np.cos(min_phi), np.cos(max_phi)],
            [np.sin(min_phi), np.sin(max_phi)]
        ]).T, eigvecs.T)
        
        # Compute concentrations
        concentrations = np.linalg.lstsq(stain_vectors.T, od.reshape(-1, 3).T, rcond=None)[0]
        
        # Reconstruct with target stains
        od_normalized = np.dot(concentrations.T, target_stains)
        rgb_normalized = np.exp(-od_normalized) * 256 - 1
        
        # Ensure valid RGB range
        rgb_normalized = np.clip(rgb_normalized, 0, 255)
        
        return rgb_normalized.reshape(patch.shape).astype(np.uint8)
    
    return normalize_patch


def extract_tissue_patches(svs_path, xml_path, patch_size=224, level=0, normalize=False, reference_patch=None):
    """
    Extract patches from the slide that overlap with annotations and contain tissue.
    
    Args:
        svs_path: Path to the .svs slide file
        xml_path: Path to the annotation .xml file
        patch_size: Size of patches to extract (default: 256)
        level: Pyramid level to extract from (default: 0, highest resolution)
        normalize: Whether to apply Macenko normalization (default: False)
        reference_patch: Optional reference image for normalization
    
    Returns:
        patches: List of (patch_size x patch_size x 3) numpy arrays
        coordinates: List of (x, y) coordinates where patches were extracted
    """
    # Get normalizer if requested
    normalizer = get_macenko_normalizer(reference_patch) if normalize else None
    
    # Read annotations
    regions, mpp = read_xml_annotations(xml_path)
    
    # Open the slide
    slide = openslide.OpenSlide(str(svs_path))
    
    # Get level dimensions and downsampling factor
    level_dims = slide.level_dimensions[level]
    level_downsample = slide.level_downsamples[level]
    
    # Calculate grid dimensions
    n_rows = int(level_dims[1] // patch_size)
    n_cols = int(level_dims[0] // patch_size)
    
    # Initialize lists to store patches and their coordinates
    patches = []
    coordinates = []
    
    # Scale factor for annotation coordinates
    scale_x = 1.0 / level_downsample
    scale_y = 1.0 / level_downsample
    
    # Iterate through grid
    for row in range(n_rows):  # tqdm(range(n_rows), desc="Extracting patches"):
        for col in range(n_cols):
            x = col * patch_size
            y = row * patch_size
            
            # Skip if patch doesn't intersect with any annotation
            if not is_patch_in_annotation(x, y, patch_size, regions, scale_x, scale_y):
                continue
            
            # Extract patch
            patch = np.array(slide.read_region(
                location=(int(x * level_downsample), int(y * level_downsample)),
                level=level,
                size=(patch_size, patch_size)
            ))
            
            # Convert from RGBA to RGB
            patch = patch[:, :, :3]
            
            # Check if patch contains tissue
            if is_tissue(patch):
                if normalize:
                    patch = normalizer(patch)
                patches.append(patch)
                coordinates.append((x, y))
    
    return patches, coordinates


def analyze_patches_with_instanseg(patches, svs_path):
    """
    Analyze patches using InstanSeg to detect nuclei.
    """
    # Initialize InstanSeg with brightfield model
    pixel_size = instanseg_brightfield.read_image(svs_path)[1]
    
    nuclei_counts = []
    labeled_outputs = []
    
    slide_name = Path(svs_path).stem
    
    # Process each patch
    for i, patch in tqdm(enumerate(patches), desc=f"Processing patches from slide {slide_name}"):
        # Run InstanSeg on the patch
        labeled_output = instanseg_brightfield.eval_small_image(patch, pixel_size)
        
        # Get the segmentation mask (assuming it's the first element)
        if isinstance(labeled_output, (list, tuple)):
            mask = labeled_output[0]  # Take first element if multiple outputs
        else:
            mask = labeled_output
            
        # Count unique labels (excluding background label 0)
        nuclei_count = len(np.unique(mask)) - 1
        nuclei_counts.append(nuclei_count)
        labeled_outputs.append(mask)
    
    # Calculate statistics
    nuclei_stats = {
        'mean_count': np.mean(nuclei_counts),
        'median_count': np.median(nuclei_counts),
        'total_count': sum(nuclei_counts),
        'counts_per_patch': nuclei_counts
    }
    return nuclei_stats, labeled_outputs


def train_tumor_classifier(slide_dnn_paths, tissue_annotation_paths, tumor_annotation_paths, patch_size=224, n_samples=100, build_cache=False, cache_dir="cached_patches"):
    """
    Train a random forest classifier using tumor annotations as positive samples
    and non-overlapping DNN regions as negative samples.
    
    Args:
        slide_dnn_paths: List of (svs_path, xml_path) for the DNN slide annotations
        tissue_annotation_paths: List of (svs_path, xml_path) for the tissue annotations
        tumor_annotation_paths: List of (svs_path, xml_path) for the tumor annotations
        patch_size: Size of patches to extract (default: 224)
        n_samples: Number of patches to sample per class (tumor/non-tumor)
    
    Returns:
        classifier: Trained RandomForestClassifier
        accuracy: Accuracy score on the held-out test slide
    """
    # Hold out last slide for testing
    train_slides = slide_dnn_paths[:-1]
    train_tissue = tissue_annotation_paths[:-1]
    train_tumor = tumor_annotation_paths[:-1]
    
    test_slide = slide_dnn_paths[-1]
    test_tissue = tissue_annotation_paths[-1]
    test_tumor = tumor_annotation_paths[-1]
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Extract positive samples from tumor annotations for training slides
    if build_cache:
        tumor_patche_names, tumor_coord_names = [], []
        for idx, (svs_tumor, xml_tumor) in tqdm(enumerate(zip(train_slides, train_tumor)), desc="Extracting patches from tumor annotations", total=len(train_slides)):
            tp, tc = extract_tissue_patches(svs_tumor, xml_tumor, patch_size=patch_size, level=0)
            slide_name = str(svs_tumor).split(os.path.sep)[-1].split(".")[0]
            patch_name = f"tumor_patches_{slide_name}_{idx}.npy"
            coord_name = f"tumor_coords_{slide_name}_{idx}.npy"
            np.save(os.path.join(cache_dir, patch_name), tp)
            np.save(os.path.join(cache_dir, coord_name), tc)
            tumor_patche_names.append(patch_name)
            tumor_coord_names.append(coord_name)
        
        # Extract patches from DNN slides for training
        dnn_patche_names, dnn_coord_names = [], []
        for idx, (svs_dnn, xml_dnn) in tqdm(enumerate(zip(train_slides, train_tissue)), desc="Extracting patches from DNN slides", total=len(train_slides)):
            dp, dc = extract_tissue_patches(svs_dnn, xml_dnn, patch_size=patch_size, level=0)
            slide_name = str(svs_dnn).split(os.path.sep)[-1].split(".")[0]
            patch_name = f"dnn_patches_{slide_name}_{idx}.npy"
            coord_name = f"dnn_coords_{slide_name}_{idx}.npy"
            np.save(os.path.join(cache_dir, patch_name), dp)
            np.save(os.path.join(cache_dir, coord_name), dc)
            dnn_patche_names.append(patch_name)
            dnn_coord_names.append(coord_name)
    else:
        tumor_patche_names, tumor_coord_names = [], []
        for idx, (svs_tumor, xml_tumor) in tqdm(enumerate(zip(train_slides, train_tumor)), desc="Extracting patches from tumor annotations", total=len(train_slides)):
            slide_name = str(svs_tumor).split(os.path.sep)[-1].split(".")[0]
            patch_name = f"tumor_patches_{slide_name}_{idx}.npy"
            coord_name = f"tumor_coords_{slide_name}_{idx}.npy"
            tumor_patche_names.append(patch_name)
            tumor_coord_names.append(coord_name)
        dnn_patche_names, dnn_coord_names = [], []
        for idx, (svs_dnn, xml_dnn) in tqdm(enumerate(zip(train_slides, train_tissue)), desc="Extracting patches from DNN slides", total=len(train_slides)):
            slide_name = str(svs_dnn).split(os.path.sep)[-1].split(".")[0]
            patch_name = f"dnn_patches_{slide_name}_{idx}.npy"
            coord_name = f"dnn_coords_{slide_name}_{idx}.npy"
            dnn_patche_names.append(patch_name)
            dnn_coord_names.append(coord_name)

    # Load patches and coordinates
    tumor_patches = []
    tumor_coords = []
    dnn_patches = []
    dnn_coords = []
    import pdb;pdb.set_trace()
    for patch_name, coord_name in tqdm(zip(tumor_patches, tumor_coords), desc="Loading tumor patches and coordinates", total=len(tumor_patches)):
        tumor_patches.append(np.load(os.path.join(cache_dir, patch_name)))
        tumor_coords.append(np.load(os.path.join(cache_dir, coord_name)))
    for patch_name, coord_name in tqdm(zip(dnn_patches, dnn_coords), desc="Loading DNN patches and coordinates", total=len(dnn_patches)):
        dnn_patches.append(np.load(os.path.join(cache_dir, patch_name)))
        dnn_coords.append(np.load(os.path.join(cache_dir, coord_name)))
    non_tumor_patches = []
    non_tumor_coords = []
    for tumor_coord, dnn_coord in tqdm(zip(tumor_coords, dnn_coords), desc="Loading non-tumor patches and coordinates", total=len(tumor_coords)):
        tumor_coords_set = {(x, y) for x, y in tumor_coord}
        dnn_coords_set = {(x, y) for x, y in dnn_coord}
        import pdb;pdb.set_trace()
        non_tumor_coords_set = dnn_coords_set - tumor_coords_set
        for coord in non_tumor_coords_set:
            non_tumor_patches.append(dnn_patches[i])
            non_tumor_coords.append(coord)  
    
    # Find coordinates that are in DNN but not in tumor annotations
    non_tumor_coords_set = dnn_coords_set - tumor_coords_set
    
    # Get the non-tumor patches using the coordinate differences
    non_tumor_patches = []
    non_tumor_coords = []
    for i, coord in enumerate(dnn_coords):
        if coord in non_tumor_coords_set:
            non_tumor_patches.append(dnn_patches[i])
            non_tumor_coords.append(coord)
    
    # Get DINO embeddings for tumor patches
    tumor_embs = []
    with torch.no_grad():
        for patch in tqdm(tumor_patches, desc="Extracting embeddings for tumor patches"):
            x = torch.from_numpy(patch).float() / 255.
            z = ((x - mu[None, None]) / std[None, None]).float()
            emb = dino_model(z.permute(2, 0, 1)[None].float())
            tumor_embs.append(emb.cpu().numpy())
    tumor_embs = np.asarray(tumor_embs)
    
    # Get DINO embeddings for non-tumor patches
    non_tumor_embs = []
    with torch.no_grad():
        for patch in tqdm(non_tumor_patches, desc="Extracting embeddings for non-tumor patches"):
            x = torch.from_numpy(patch).float() / 255.
            z = ((x - mu[None, None]) / std[None, None]).float()
            emb = dino_model(z.permute(2, 0, 1)[None].float())
            non_tumor_embs.append(emb.cpu().numpy())
    non_tumor_embs = np.asarray(non_tumor_embs)
    
    # Sample equal numbers from each class for training
    n_samples = min(n_samples, len(tumor_embs), len(non_tumor_embs))
    tumor_indices = np.random.choice(len(tumor_embs), n_samples, replace=False)
    non_tumor_indices = np.random.choice(len(non_tumor_embs), n_samples, replace=False)
    
    # Combine samples and create labels for training
    X_train = np.concatenate([
        tumor_embs[tumor_indices].squeeze(),
        non_tumor_embs[non_tumor_indices].squeeze()
    ])
    y_train = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
    
    # Extract test patches from held-out slide
    test_tumor_patches, test_tumor_coords = extract_tissue_patches(test_slide, test_tumor, patch_size=patch_size, level=0)
    test_dnn_patches, test_dnn_coords = extract_tissue_patches(test_slide, test_tissue, patch_size=patch_size, level=0)
    
    # Get non-overlapping patches for test set
    test_tumor_coords_set = {(x, y) for x, y in test_tumor_coords}
    test_dnn_coords_set = {(x, y) for x, y in test_dnn_coords}
    test_non_tumor_coords_set = test_dnn_coords_set - test_tumor_coords_set
    
    test_non_tumor_patches = []
    for i, coord in enumerate(test_dnn_coords):
        if coord in test_non_tumor_coords_set:
            test_non_tumor_patches.append(test_dnn_patches[i])
    
    # Get embeddings for test patches
    test_tumor_embs = []
    test_non_tumor_embs = []
    with torch.no_grad():
        for patch in tqdm(test_tumor_patches, desc="Extracting embeddings for test tumor patches"):
            x = torch.from_numpy(patch).float() / 255.
            z = ((x - mu[None, None]) / std[None, None]).float()
            emb = dino_model(z.permute(2, 0, 1)[None].float())
            test_tumor_embs.append(emb.cpu().numpy())
            
        for patch in tqdm(test_non_tumor_patches, desc="Extracting embeddings for test non-tumor patches"):
            x = torch.from_numpy(patch).float() / 255.
            z = ((x - mu[None, None]) / std[None, None]).float()
            emb = dino_model(z.permute(2, 0, 1)[None].float())
            test_non_tumor_embs.append(emb.cpu().numpy())
    
    test_tumor_embs = np.asarray(test_tumor_embs)
    test_non_tumor_embs = np.asarray(test_non_tumor_embs)
    
    # Create test set
    X_test = np.concatenate([
        test_tumor_embs.squeeze(),
        test_non_tumor_embs.squeeze()
    ])
    y_test = np.concatenate([
        np.ones(len(test_tumor_embs)),
        np.zeros(len(test_non_tumor_embs))
    ])
    
    # Train random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Calculate accuracy on held-out test slide
    accuracy = clf.score(X_test, y_test)
    
    return clf, accuracy


# Example usage
if __name__ == "__main__":
    xml_path = Path("moffit_data/Pass_Slides_DNN/2-P.xml")
    svs_path = xml_path.with_suffix('.svs')
    output_path = str(xml_path.with_suffix('.png'))
    
    # Run with nuclei analysis
    nuclei_stats = plot_annotations_on_slide(
        svs_path=svs_path,
        xml_path=xml_path,
        output_path=output_path,
        analyze_nuclei=True,
        debug_nuclei=True
    )
    
    # Train tumor classifier
    classifier, accuracy = train_tumor_classifier(svs_path, xml_path)
    print(f"Tumor classifier accuracy: {accuracy:.2f}")
