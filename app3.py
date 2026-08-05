from flask import Flask, render_template, request
"""render_template → loads HTML files like index.html
request → handles uploaded files / form data from user"""
import numpy as np
import os
import uuid                 
import shutil        
import subprocess   #subprocess is used to run external commands from Python.
import tempfile                 #is used to create temporary folders or files.
import matplotlib   #for Flask/server apps. Do not open a window. Just create the image in memory or save it.

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from io import BytesIO   #creates an in-memory file. f saving the result image into RAM memory.
import base64            #converts image bytes into text format. display an image directly inside HTML
import nibabel as nib    #is used to read and write medical image files, especially .nii.gz files.
import torch

print(torch.cuda.is_available())
print(torch.__version__)

app = Flask(__name__)    #creates your Flask web application object.  Flask(__name__) tells Flask where the app is located so it can find templates and static files.

# -------------------------------------------------
# Official nnU-Net settings
# -------------------------------------------------

# IMPORTANT:
# This should be the parent nnUNet_results folder.
# It should contain Dataset001_BraTS2021 inside it.
NNUNET_RESULTS = r"D:/Uni/Seven/Project/Final_Project_V1/NNUnet_Brain_Prediction/nnUNet_results"

os.environ["nnUNet_results"] = NNUNET_RESULTS
os.environ["nnUNet_raw"] = os.path.abspath("nnUNet_raw_app")
os.environ["nnUNet_preprocessed"] = os.path.abspath("nnUNet_preprocessed_app")

os.makedirs(os.environ["nnUNet_raw"], exist_ok=True)
os.makedirs(os.environ["nnUNet_preprocessed"], exist_ok=True)

DATASET_ID = "Dataset001_BraTS2021"
CONFIGURATION = "3d_fullres"
FOLD = "0"
CHECKPOINT_NAME = "checkpoint_best.pth"

MODEL_CHECKPOINT_PATH = os.path.join(
    NNUNET_RESULTS,
    "Dataset001_BraTS2021",
    "nnUNetTrainer__nnUNetPlans__3d_fullres",
    "fold_0",
    "checkpoint_best.pth"
)

print("Checkpoint exists:", os.path.exists(MODEL_CHECKPOINT_PATH))
print("Checkpoint path:", MODEL_CHECKPOINT_PATH)

if not os.path.exists(MODEL_CHECKPOINT_PATH):
    print("WARNING: checkpoint_best.pth not found at:")
    print(MODEL_CHECKPOINT_PATH)
else:
    print("Official nnU-Net checkpoint found:")
    print(MODEL_CHECKPOINT_PATH)


# -------------------------------------------------
# Flask folders
# -------------------------------------------------

UPLOAD_FOLDER = os.path.abspath("static/uploads")
TEMP_NNUNET_FOLDER = os.path.abspath("static/nnunet_temp")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_NNUNET_FOLDER, exist_ok=True)

#This line stores your upload folder path inside the Flask app configuration.
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER   


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def remap_brats_labels(seg_data):
    """
    BraTS original labels:
    0 = background
    1 = necrotic / non-enhancing tumor
    2 = edema
    4 = enhancing tumor

    nnU-Net labels used in your training:
    0, 1, 2, 3
    """
    fixed = np.zeros_like(seg_data, dtype=np.uint8) #This creates a new array/segmentation mask with the same shape as seg_data, but all values are 0.
    fixed[seg_data == 1] = 1
    fixed[seg_data == 2] = 2
    fixed[seg_data == 3] = 3
    fixed[seg_data == 4] = 3
    return fixed


def load_nifti_volume(file_path):
    return nib.load(file_path).get_fdata()
    #loads a NIfTI MRI file and returns its data as a 3d NumPy array.


def load_ground_truth_volume(seg_path):
    seg_data = nib.load(seg_path).get_fdata().astype(np.uint8)  #convert values in integer NumPy array.
    return remap_brats_labels(seg_data)


def choose_best_slice(gt_mask, pred_mask=None):
    """
    Select a slice where tumor is visible.
    First priority: ground truth tumor slice.
    Second priority: prediction tumor slice.
    Third priority: middle slice.
    """
    tumor_slices = np.where(np.sum(gt_mask > 0, axis=(0, 1)) > 0)[0]

    if len(tumor_slices) > 0:
        return tumor_slices[len(tumor_slices) // 2]

    if pred_mask is not None:
        pred_slices = np.where(np.sum(pred_mask > 0, axis=(0, 1)) > 0)[0]
        if len(pred_slices) > 0:
            return pred_slices[len(pred_slices) // 2]

    return gt_mask.shape[2] // 2


def run_official_nnunet_prediction(flair_path, t1_path, t1ce_path, t2_path, case_id):
    """
    Runs official nnUNetv2_predict on one uploaded patient.
    This creates temporary nnU-Net input files:
        case_id_0000.nii.gz = FLAIR
        case_id_0001.nii.gz = T1
        case_id_0002.nii.gz = T1ce
        case_id_0003.nii.gz = T2
    """

    work_dir = tempfile.mkdtemp(prefix=f"{case_id}_", dir=TEMP_NNUNET_FOLDER)

    input_dir = os.path.join(work_dir, "input")
    output_dir = os.path.join(work_dir, "output")

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    shutil.copy(flair_path, os.path.join(input_dir, f"{case_id}_0000.nii.gz"))
    shutil.copy(t1_path, os.path.join(input_dir, f"{case_id}_0001.nii.gz"))
    shutil.copy(t1ce_path, os.path.join(input_dir, f"{case_id}_0002.nii.gz"))
    shutil.copy(t2_path, os.path.join(input_dir, f"{case_id}_0003.nii.gz"))

    command = [
    "nnUNetv2_predict",
    "-i", input_dir,
    "-o", output_dir,
    "-d", DATASET_ID,
    "-c", CONFIGURATION,
    "-f", FOLD,
    "-chk", CHECKPOINT_NAME,
    "-device", "cpu",
    "--disable_tta"
]

    result = subprocess.run(
        command,
        capture_output=True,  #→ save command output and error messages        
        text=True             # → return output as normal text
    )

    #This checks if the command failed. returncode =0 is success
    if result.returncode != 0:  
        raise RuntimeError(
            "nnU-Net prediction failed.\n\n"
            f"STDOUT:\n{result.stdout}\n\n"    #usually shows normal command output.
            f"STDERR:\n{result.stderr}"        #show error         
        )

    pred_path = os.path.join(output_dir, f"{case_id}.nii.gz") 

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    return pred_path, work_dir


def create_overlay(image, mask, is_prediction=False):
    if is_prediction:
        colors = {
            1: [255, 80, 80], #light red
            2: [80, 255, 80], #light Green
            3: [80, 80, 255]  #Bluw
        }
    else:
        colors = {
            1: [255, 0, 0],  #strong red
            2: [0, 255, 0],
            3: [0, 0, 255]
        }

    overlay = np.zeros((*image.shape, 3), dtype=np.uint8)

    img_norm = (
        (image - image.min()) /
        (image.max() - image.min() + 1e-6) * 255
    ).astype(np.uint8)

    overlay[:, :, 0] = img_norm
    overlay[:, :, 1] = img_norm
    overlay[:, :, 2] = img_norm

    for class_id, color in colors.items():
        overlay[mask == class_id] = color

    return overlay


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    return img_base64


def predict_segmentation(flair_path, t1_path, t1ce_path, t2_path, seg_path, case_id):
    """
    Official 3D nnU-Net prediction.
    This function predicts full 3D mask and then selects one slice for display.
    """

    pred_path, work_dir = run_official_nnunet_prediction(
        flair_path,
        t1_path,
        t1ce_path,
        t2_path,
        case_id
    )

    try:
        flair = load_nifti_volume(flair_path)
        t1 = load_nifti_volume(t1_path)
        t1ce = load_nifti_volume(t1ce_path)
        t2 = load_nifti_volume(t2_path)

        gt_mask = load_ground_truth_volume(seg_path)
        pred_mask = nib.load(pred_path).get_fdata().astype(np.uint8)

        slice_idx = choose_best_slice(gt_mask, pred_mask)

        flair_slice = flair[:, :, slice_idx]
        t1_slice = t1[:, :, slice_idx]
        t1ce_slice = t1ce[:, :, slice_idx]
        t2_slice = t2[:, :, slice_idx]

        gt_slice = gt_mask[:, :, slice_idx]
        pred_slice = pred_mask[:, :, slice_idx]

        print("Prediction labels:", np.unique(pred_mask))
        print("Selected slice:", slice_idx)

        return pred_slice, gt_slice, flair_slice, t1_slice, t1ce_slice, t2_slice

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# -------------------------------------------------
# Flask route
# -------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        flair_file = request.files.get("flair")
        t1_file = request.files.get("t1")
        t1ce_file = request.files.get("t1ce")
        t2_file = request.files.get("t2")
        seg_file = request.files.get("seg")

        if not all([flair_file, t1_file, t1ce_file, t2_file, seg_file]):
            return render_template(
                "index.html",
                error="Please upload all 5 files: FLAIR, T1, T1ce, T2, and segmentation mask."
            )

        unique_id = uuid.uuid4().hex[:8]
        case_id = f"case_{unique_id}"

        flair_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{case_id}_flair.nii.gz")
        t1_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{case_id}_t1.nii.gz")
        t1ce_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{case_id}_t1ce.nii.gz")
        t2_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{case_id}_t2.nii.gz")
        seg_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{case_id}_seg.nii.gz")

        flair_file.save(flair_path)
        t1_file.save(t1_path)
        t1ce_file.save(t1ce_path)
        t2_file.save(t2_path)
        seg_file.save(seg_path)

        uploaded_paths = [
            flair_path,
            t1_path,
            t1ce_path,
            t2_path,
            seg_path
        ]

        try:
            pred_mask, gt_mask, flair_slice, t1_slice, t1ce_slice, t2_slice = predict_segmentation(
                flair_path,
                t1_path,
                t1ce_path,
                t2_path,
                seg_path,
                case_id
            )

            overlay_gt = create_overlay(flair_slice, gt_mask, is_prediction=False)
            overlay_pred = create_overlay(flair_slice, pred_mask, is_prediction=True)

            fig, axes = plt.subplots(2, 4, figsize=(18, 10))

            axes[0, 0].imshow(flair_slice, cmap="gray")
            axes[0, 0].set_title("FLAIR")
            axes[0, 0].axis("off")

            axes[0, 1].imshow(t1_slice, cmap="gray")
            axes[0, 1].set_title("T1")
            axes[0, 1].axis("off")

            axes[0, 2].imshow(t1ce_slice, cmap="gray")
            axes[0, 2].set_title("T1ce")
            axes[0, 2].axis("off")

            axes[0, 3].imshow(t2_slice, cmap="gray")
            axes[0, 3].set_title("T2")
            axes[0, 3].axis("off")

            axes[1, 0].imshow(gt_mask, cmap="tab10", vmin=0, vmax=3)
            axes[1, 0].set_title("Ground Truth Mask")
            axes[1, 0].axis("off")

            axes[1, 1].imshow(pred_mask, cmap="tab10", vmin=0, vmax=3)
            axes[1, 1].set_title("Official 3D nnU-Net Prediction")
            axes[1, 1].axis("off")

            axes[1, 2].imshow(overlay_gt)
            axes[1, 2].set_title("Ground Truth Overlay")
            axes[1, 2].axis("off")

            axes[1, 3].imshow(overlay_pred)
            axes[1, 3].set_title("Prediction Overlay")
            axes[1, 3].axis("off")

            plt.tight_layout()
            img_base64 = fig_to_base64(fig)
            plt.close(fig)

            tumor_present = np.any(pred_mask > 0)
            result_text = "TUMOR DETECTED" if tumor_present else "NO TUMOR DETECTED"

            for p in uploaded_paths:
                if os.path.exists(p):
                    os.remove(p)

            return render_template(
                "index.html",
                result_image=img_base64,
                result_text=result_text
            )

        except Exception as e:
            for p in uploaded_paths:
                if os.path.exists(p):
                    os.remove(p)

            return render_template(
                "index.html",
                error=f"Error processing files: {str(e)}"
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)