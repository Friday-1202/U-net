# U-net

### Qualitative Results

**Observations**:
- **Large areas** (background, roof, green field) are segmented correctly with smooth boundaries.
- **Small objects** (umbrella, sedan, truck) are detected but boundaries are slightly rough.
- **Confusion** exists between similar classes, e.g., `dirt_motor_road` vs. `paved_motor_road`.
- **Rare classes** (`class_25`, `umbrella`) show low IoU due to limited training samples.

---

## Discussion

### Why the Model Performs Decently

- **U‑Net architecture** is well suited for aerial images: skip connections retain fine spatial details needed for boundary delineation.  
- **Combined loss** (Cross‑Entropy + Dice) mitigates class imbalance, especially for small objects like vehicles.  
- **Rescaling to 0.15** trades off some high‑frequency detail for feasible GPU memory, yet overall shapes remain correct.

### Hyperparameter Reflections

| Hyperparameter | Impact |
|----------------|--------|
| **Scale = 0.15** | Reduces memory but may hurt small‑object IoU. Increasing to 0.25 could improve performance but doubles training time and risks OOM. |
| **Epochs > 12** | Validation Dice plateaus after epoch 8; further training leads to overfitting (training loss keeps decreasing but validation Dice stagnates). |
| **Batch size = 1** | Limits gradient stability; gradient accumulation could be used, but was not implemented here. |
| **Learning rate = 1e‑4** | AdamW with default betas worked well; no need for LR scheduling. |

### Possible Improvements

| Priority | Improvement | Expected Benefit |
|----------|-------------|------------------|
| High | **Data augmentation** (random rotation, flip, colour jitter) | Better generalisation, especially for small/rare classes |
| Medium | **Lighter backbone** (e.g., MobileNet‑UNet) | Allows larger batch size or higher resolution |
| Low | **Post‑processing** (CRF / conditional random fields) | Refines boundaries, removes small isolated regions |
| Low | **Class‑balanced sampling** | Boosts IoU for under‑represented classes (umbrella, truck, class_25) |

### Error Sources

1. **Class imbalance**: `umbrella` and `class_25` appear very rarely → low IoU.  
2. **Similar textures**: Dirt vs. paved roads are easily confused, especially in shadows.  
3. **Resolution loss**: Down‑scaling to 15% loses fine details of small objects (e.g., car boundaries).  
4. **No data augmentation**: The model sees each training image only once per epoch in its original orientation, limiting invariance.

---

## Conclusions

1. **U‑Net achieves a solid 0.619 mean IoU** on the AMtown02 test set, with pixel accuracy of 93.1%.  
2. **Large‑area classes** (roof, green field, background) are segmented very accurately (IoU > 0.76).  
3. **Small and rare objects** remain challenging (IoU as low as 0.22), suggesting the need for data augmentation or higher input resolution.  
4. **The pipeline is fully reproducible** and runs end‑to‑end in Google Colab, using the original U‑Net repository.

### Recommendations for Future Work

| Action | Priority |
|--------|----------|
| Implement online data augmentation (random rotation, flip, colour jitter) | **High** |
| Increase `SCALE` to 0.25 if GPU memory permits (or use gradient accumulation) | **Medium** |
| Add post‑processing CRF to refine object boundaries | **Low** |
| Experiment with focal loss to further address class imbalance | **Low** |

---

## References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U‑Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*.  
2. Milesial (GitHub). *Pytorch‑UNet*. https://github.com/milesial/Pytorch-UNet  
3. AMtown dataset – part of the AAE5303 course materials, The Hong Kong Polytechnic University.  
4. Paszke, A. et al. (2019). PyTorch: An Imperative Style, High‑Performance Deep Learning Library. *NeurIPS*.

---

## Appendix

### Files

- `segment_ass_xhr.py` – Full Colab notebook (data preparation, training, evaluation)  
- `training_history.csv` – Loss and validation Dice per epoch  
- `unet_structure.png` – Diagram of the U‑Net architecture  
- `amtown02_evaluation_report.json` – Detailed test metrics and confusion matrix  
- `cmap.py` – Colour‑to‑ID mapping for the AMtown labels  

### Hardware & Software

- **Platform**: Google Colab (GPU: NVIDIA T4 / V100)  
- **Framework**: PyTorch 2.x, Torchvision  
- **Libraries**: NumPy, PIL, Matplotlib, tqdm, scikit‑image (for evaluation)

---

**AAE5303 – Robust Control Technology in Low‑Altitude Aerial Vehicle**

**The Hong Kong Polytechnic University – February 2026**
