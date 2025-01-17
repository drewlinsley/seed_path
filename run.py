import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms
from src.dataloaders import Classification
import timm
import schedulefree
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler
from src import utils
from glob import glob


def get_sampler(labels):
    unique_labels = np.unique(labels)
    # TODO: how to handle this when there's no dead cells (0) in the training set?
    class_weights = compute_class_weight("balanced", classes=unique_labels, y=labels)
    class_weights = torch.from_numpy(class_weights).float()
    samples_weight_train = np.asarray([class_weights[t] for t in labels])
    samples_weight_train = torch.from_numpy(samples_weight_train).double()
    sampler = WeightedRandomSampler(samples_weight_train, len(samples_weight_train))
    return sampler, class_weights


# Prep data
df = pd.read_excel("moffit_data/DNN_STAR_PASS_FAIL_SENT_TO_DREW.xlsx")

# Quantize DNA
quantized_DNA = utils.quantize_DNA(df.DNA.values)

# Get images and labels
images = glob(os.path.join("moffit_data", "*", "*.svs"))

# Configuration
epochs = 1000
tr_bs = 16  # 48  # Training batch size
te_bs = 50  # 48  # Testing batch size
timm_model = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
# timm_model = "convnext_base.clip_laion2b_augreg_ft_in1k"
# timm_model = "vit_large_patch14_dinov2.lvd142m"
# timm_model = "convnext_xxlarge.clip_laion2b_soup_ft_in1k"
quantization = 64  # How many levels in the reconstructed image
alpha = "balanced"  # Weight for the image next loss. If "balanced", then the image loss is balanced with binary crossentropy

new_lr = 1e-3  # 3e-4
ckpts = "soothsayer_aug_ckpts"
# "fastvit_sa12.apple_dist_in1k"
# HW = 518
HW = 448
num_workers_train = 16
num_workers_val = 16
weight_train_loss = False
weight_val_loss = False
boost_data = True
compile_model = False
balanced_training = False
balanced_validation = True
restore_ckpt = None
use_cutmix = False
eval_frequency = 32  # Evaluate every K steps, set to None to evaluate per epoch
np.random.seed(42)
minv = 0
maxv = 2 ** 16 - 1

# Prepare environment and model
# torch.distributed.init_process_group(backend='gloo')
accelerator, device, tqdm, TIMM = utils.prepare_env(timm_model)
train_trans = transforms.Compose([
    utils.RandomRotate90(),
    # transforms.RandomCrop(HW),
    transforms.RandomResizedCrop(HW, scale=(0.8, 1.3)),
    # transforms.CenterCrop(HW),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5),
    # transforms.RandomAutocontrast(),
    transforms.RandomAffine(degrees=0, translate=(0.1, .3), shear=(.0, 0.1)),
    # transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 7)),
    # transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 7)),
    # transforms.GaussianNoise(clip=False),
    # transforms.ToTensor(),
])

val_trans = transforms.Compose([
    transforms.CenterCrop(HW),
    # transforms.Resize(HW),
    # transforms.ToTensor(),
])

# Find correct alpha
if alpha == "balanced":
    class_weight = np.log(2)
    image_weight = np.log(quantization)
    alpha = class_weight / image_weight

# Load and preprocess data
with accelerator.main_process_first():
    print("Started loading data")
    data = np.load(ddir, allow_pickle=True)

    cells = data["file_pairs"]
    labels = data["label_pairs"].astype(int)
    masks = data["mask_pairs"]
    panels = data["panel_pairs"]
    wells = data["well_pairs"]
    # gedi_pairs = data["gedi_pairs"]
    gedi_pairs = labels
    stim_info_pairs = data["stim_info_pairs"]

    del data.f
    data.close()

    # Compute normalized timesteps
    timesteps = np.asarray([int(re.search(t_number_pattern, str(f)).group(1)) for f in cells])
    timesteps = (timesteps - timesteps.min()) / (timesteps.max() - timesteps.min())

    # Reduce to specific panels
    if panel_sel is not None:
        k_panel = np.isin(panels, panel_sel)
        cells = cells[k_panel]
        wells = wells[k_panel]
        panels = panels[k_panel]
        labels = labels[k_panel]
        masks = masks[k_panel]
        stim_info_pairs = stim_info_pairs[k_panel]
        gedi_pairs = gedi_pairs[k_panel]
        timesteps = timesteps[k_panel]

    # Zscore stim_info
    stim_info_pairs = (stim_info_pairs - stim_info_pairs.min()) / (stim_info_pairs.max() - stim_info_pairs.min())
    stim_info_pairs = 2 * (stim_info_pairs - 0.5)
    # stim_info_pairs = (stim_info_pairs - stim_info_pairs.mean()) / stim_info_pairs.std()

    # Split into train/test
    # Our approach is to always include train_well_names in training. Then add
    # a proportion of the remaining wells into training as well.
    val_idx = np.isin(wells, test_well_names)
    train_idx = ~val_idx
    train_wells = wells[train_idx]
    train_cells = cells[train_idx]
    train_panels = panels[train_idx]
    train_labels = labels[train_idx]
    train_masks = masks[train_idx]
    train_stim_info = stim_info_pairs[train_idx]
    train_gedi = gedi_pairs[train_idx]
    train_timesteps = timesteps[train_idx]
    val_cells = cells[val_idx]
    val_wells = wells[val_idx]
    val_panels = panels[val_idx]
    val_labels = labels[val_idx]
    val_masks = masks[val_idx]
    val_stim_info = stim_info_pairs[val_idx]
    val_gedi = gedi_pairs[val_idx]
    val_timesteps = timesteps[val_idx]
    np.random.seed(42)
    if balanced_training:
        pos_idx = np.where(train_labels == 1)[0]
        neg_idx = np.where(train_labels == 0)[0]
        neg_idx = neg_idx[np.random.permutation(len(neg_idx))[:len(pos_idx)]]
        train_idx = np.concatenate([pos_idx, neg_idx])
        train_cells = train_cells[train_idx]
        train_panels = train_panels[train_idx]
        train_wells = train_wells[train_idx]
        train_labels = train_labels[train_idx]
        train_masks = train_masks[train_idx]
        train_stim_info = train_stim_info[train_idx]
        train_gedi = train_gedi[train_idx]
        train_timesteps = train_timesteps[train_idx]

    # if balanced_validation:
    #     # Take equal #s of live/dead from each subject
    #     min_labels = val_labels.sum()
    #     pos = np.where(val_labels == 1)[0]
    #     neg = np.where(val_labels == 0)[0]
    #     pos = pos[np.random.permutation(len(pos))[:min_labels]]
    #     neg = neg[np.random.permutation(len(neg))[:min_labels]]
    #     val_idx = np.concatenate([pos, neg])
    #     val_cells = val_cells[val_idx]
    #     val_wells = val_wells[val_idx]
    #     val_panels = val_panels[val_idx]
    #     val_labels = val_labels[val_idx]
    #     val_masks = val_masks[val_idx]
    #     val_stim_info = val_stim_info[val_idx]
    #     val_gedi = val_gedi[val_idx]
    #     val_timesteps = val_timesteps[val_idx]

    # Create datasets
    train_dataset = ClassificationSurvivalSTImageClassificationExtra(
        files=train_cells,
        labels=train_labels,
        masks=train_masks,
        timesteps=train_timesteps,
        stim_info=train_stim_info,
        gedi=train_gedi,
        quantization=quantization - 1,
        mu=TIMM["mean"],
        std=TIMM["std"],
        minv=minv,
        maxv=maxv,
        transform=train_trans
    )
    val_dataset = ClassificationSurvivalSTImageClassificationExtra(
        files=val_cells,
        labels=val_labels,
        masks=val_masks,
        timesteps=val_timesteps,
        stim_info=val_stim_info,
        gedi=val_gedi,
        quantization=quantization - 1,
        mu=TIMM["mean"],
        std=TIMM["std"],
        minv=minv,
        maxv=maxv,
        transform=val_trans
    )
    print("Finished loading data")

# Create data loaders
train_sampler, train_class_weights = get_sampler(train_labels)
val_sampler, val_class_weights = get_sampler(val_labels)
print("Building dataloaders")
if boost_data:
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=tr_bs,
        sampler=train_sampler,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers_train
    )
else:
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=tr_bs,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers_train
    )

if balanced_validation:
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=te_bs, sampler=val_sampler, drop_last=False, num_workers=num_workers_val)
else:
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=te_bs, shuffle=False, drop_last=False, num_workers=num_workers_val)
print("Including {}/{} images/batches in training and {}/{} images/batches in validation".format(len(train_loader) * tr_bs, len(train_loader), len(val_loader) * te_bs,  len(val_loader)))

# Initialize model
print("Preparing models")
model = timm.create_model(timm_model, pretrained=True, num_classes=2, in_chans=4)
if restore_ckpt is not None:
    print("Restoring from ckpt: {}".format(restore_ckpt))
    ckpt_weights = torch.load(restore_ckpt, map_location=torch.device('cpu'), weights_only=True)
    key_check = [x for x in ckpt_weights.keys()][0]
    if key_check.split(".")[0] == "module":
        ckpt_weights = {k.replace("module.", ""): v for k, v in ckpt_weights.items()}
    elif key_check.split(".")[0] == "_orig_mod":
        ckpt_weights = {k.replace("_orig_mod.", ""): v for k, v in ckpt_weights.items()}
    model.load_state_dict(ckpt_weights, strict=True)

optimizer = schedulefree.AdamWScheduleFree(model.parameters(), lr=new_lr)
os.makedirs(ckpts, exist_ok=True)

if compile_model:
    model = torch.compile(model)

# model, optimizer, train_loader, val_loader, class_weights, loss_weights = accelerator.prepare(model, optimizer, train_loader, val_loader, class_weights, loss_weights)
model, optimizer, train_loader, val_loader = accelerator.prepare(model, optimizer, train_loader, val_loader)
train_class_weights = train_class_weights.to(device)
val_class_weights = val_class_weights.to(device)

# Add additional augs
if use_cutmix:
    raise NotImplementedError("CutMix not implemented")
    cutmix = transforms.CutMix(num_classes=2)

# Replace the training loop with this updated version
best_loss = float('inf')
global_step = 0
for epoch in range(epochs):
    # Training phase
    model.train()
    if not num_workers_train:
        print("Warning: Not parallelizing dataloading. set num_workers_train > 0.")
    if accelerator.is_main_process:
        train_progress = tqdm(total=len(train_loader), desc=f"Training Epoch {epoch+1}/{epochs}")

    losses, accs = [], []
    for step, source in enumerate(train_loader):
        optimizer.zero_grad()
        images, labels, gedis = source
        preds_CCE = model(images)

        if weight_train_loss:
            loss_CCE = F.cross_entropy(preds_CCE, labels, weight=train_class_weights, reduction="none").mean()
        else:
            loss_CCE = F.cross_entropy(preds_CCE, labels)

        loss = loss_CCE
        preds_CCE = preds_CCE.argmax(dim=1).detach().cpu().numpy()
        labels = labels.cpu().numpy()
        acc = (labels == preds_CCE).mean().astype(np.float32)
        prop = labels.mean().item()

        if torch.isnan(loss):
            import pdb;pdb.set_trace()

        accelerator.backward(loss)
        optimizer.step()
        global_step += 1

        loss = loss_CCE.item()
        losses.append(loss)
        accs.append(acc)

        if accelerator.is_main_process:
            train_progress.set_postfix({"Train loss": f"{loss:.4f}", "Train acc": f"{acc:.4f}", "Train prop": f"{prop:.4f}"})
            train_progress.update()

        # Evaluate if needed
        if eval_frequency is not None and global_step % eval_frequency == 0:
            # Evaluation phase
            model.eval()
            if accelerator.is_main_process:
                data_len = len(val_loader)
                val_progress = tqdm(total=data_len, desc=f"Testing Step {global_step}")
            
            accs_array, losses_array = [], []
            with torch.no_grad():
                for val_step, source in enumerate(val_loader):
                    images, labels, gedis = source
                    preds_CCE = model(images)
                    loss_CCE = F.cross_entropy(preds_CCE, labels)
                    if weight_val_loss:
                        loss_CCE = F.cross_entropy(preds_CCE, labels, weight=val_class_weights)
                    else:
                        loss_CCE = F.cross_entropy(preds_CCE, labels)
                    loss = loss_CCE.item()

                    labels = labels.cpu().numpy()
                    preds_CCE = preds_CCE.argmax(dim=1).detach().cpu().numpy()
                    acc = (labels == preds_CCE).mean().astype(np.float32)
                    accs_array.append(acc)
                    losses_array.append(loss)

                    if accelerator.is_main_process:
                        val_progress.update()

                # Calculate average loss and save best model
                avg_loss = np.mean(losses_array)
                avg_acc = np.mean(accs_array)

                if accelerator.is_main_process:
                    if avg_loss < best_loss:
                        checkpoint_filename = os.path.join(ckpts, f'model_step_{global_step}.pth')
                        torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_filename)
                        best_loss = avg_loss
                        best_acc = avg_acc
                    val_progress.set_postfix({"Avg val loss": f"{avg_loss:.4f}", "Avg train loss": f"{np.mean(losses):.4f}", "Best val loss": f"{best_loss:.4f}", "Best val acc": f"{best_acc:.4f}"})
                    val_progress.close()
            
            # Return to training mode
            model.train()
            accelerator.wait_for_everyone()

    # Epoch-end evaluation (only if eval_frequency is None)
    if eval_frequency is None:
        # Evaluation phase
        model.eval()
        if accelerator.is_main_process:
            data_len = len(val_loader)
            val_progress = tqdm(total=data_len, desc=f"Testing Epoch {epoch+1}/{epochs}")
        
        accs_array, losses_array = [], []
        with torch.no_grad():
            for step, source in enumerate(val_loader):
                images, labels, gedis = source
                preds_CCE = model(images)
                loss_CCE = F.cross_entropy(preds_CCE, labels)
                if weight_val_loss:
                    loss_CCE = F.cross_entropy(preds_CCE, labels, weight=val_class_weights)
                else:
                    loss_CCE = F.cross_entropy(preds_CCE, labels)
                loss = loss_CCE.item()

                labels = labels.cpu().numpy()
                preds_CCE = preds_CCE.argmax(dim=1).detach().cpu().numpy()
                acc = (labels == preds_CCE).mean().astype(np.float32)
                accs_array.append(acc)
                losses_array.append(loss)

                if accelerator.is_main_process:
                    val_progress.set_postfix({"Step": f"{step}"})
                    val_progress.update()

            # Calculate average loss and save best model
            avg_loss = np.mean(losses_array)
            avg_acc = np.mean(accs_array)

            # Add to best checkpoints if it's in the top K
            if accelerator.is_main_process:
                if avg_loss < best_loss:
                    checkpoint_filename = os.path.join(ckpts, f'model_epoch_{epoch+1}.pth')
                    torch.save(accelerator.unwrap_model(model).state_dict(), checkpoint_filename)
                    best_loss = avg_loss
                    best_acc = avg_acc
                val_progress.set_postfix({"Avg val loss": f"{avg_loss:.4f}", "Avg train loss": f"{np.mean(losses):.4f}", "Best val loss": f"{best_loss:.4f}", "Best val acc": f"{best_acc:.4f}"})
                val_progress.close()
        accelerator.wait_for_everyone()
