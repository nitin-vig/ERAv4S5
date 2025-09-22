Here’s a **README.md** file you can add to your repo. It includes lots of learning‑focused comments to help anyone understand *why* the choices were made. You can save it as `README.md`.

---

````markdown
# MNIST Training with <20,000 Parameters

This repository demonstrates training a compact convolutional neural network on the MNIST dataset with **fewer than 20,000 trainable parameters**, while still achieving high accuracy. Emphasis is placed on understanding design choices, trade‑offs, and techniques for efficient learning.

---

## 📂 Repository Contents

- `MNIST_Training_in_less_than_20k_Params.ipynb` — Jupyter notebook showing the full process: model definition, training, validation, and evaluation.  
- `data/` — MNIST data is downloaded here via torchvision.  
- `README.md` — This file.  
- `requirements.txt` — Python package dependencies.  

---

## 🎯 Purpose and Learning Goals

- Learn how to **design small‑capacity models** that generalize well.  
- Understand the roles of **Batch Normalization**, **Dropout**, and **Data Augmentation**.  
- See how **optimizer choice** and **learning rate scheduling** affect convergence.  
- Explore trade‑offs between **parameter count**, **training speed**, and **accuracy**.  
- Encourage experimentation to push beyond “easy” plateaus.

---

## 🧠 Key Design Decisions & Explanations

| Component | What was done | Why it matters |
|-----------|----------------|----------------|
| **Tiny model (<20k params)** | Few convolutional filters, small FC layer | Forces careful balance: enough capacity to learn but not so much that overfitting or resource waste happens |
| **Batch Normalization (after conv layers)** | Helps stabilize & accelerate training, allows use of higher learning rates | Reduces internal covariate shift; helps gradients be more stable |
| **Dropout (light, after FC layer, after activation)** | Helps generalization without crippling the model | Dropout after ReLU in FC removes random active units, reducing co‑adaptation |
| **Data Augmentation** | Small rotations / translations (via `RandomRotation`, `RandomAffine`) | Makes model robust to small input perturbations; helps reduce overfit |
| **Optimizer & Learning Rate** | SGD with momentum, moderate LR (e.g. 0.01), with scheduling | Momentum helps smooth optimization; scheduling avoids getting stuck at accuracy plateaus |
| **Training / Validation Tracking** | Monitor both loss & accuracy over epochs | Ensures you see overfitting vs under‑fitting; decide whether to increase capacity or regularization |

---

## ⚙ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/nitin-vig/ERAv4S5.git
   cd ERAv4S5
````

2. Install dependencies (you can use `requirements.txt`):

   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:

   ```bash
   jupyter notebook MNIST_Training_in_less_than_20k_Params.ipynb
   ```

   Ensure you run all cells in order. The notebook includes sections:

   * Data loading and transforms
   * Model definition (with parameter count)
   * Training loop with optimizer & learning rate scheduler
   * Evaluation on test set
   * Experiments / tips for further improvements

---

## 🧪 Expected Results

* Test accuracy in the ballpark of **\~99%** (depending on augmentation & training settings).
* Parameter count stays under **20,000**.
* Model trains relatively quickly on standard hardware.

---

## 🔍 Tips for Pushing Further

* **Try different dropout rates** (e.g. 0.1, 0.2) or adding very light dropout after convolutional blocks.
* **Experiment with learning rate schedules**: e.g., cosine annealing, step decay, or manual decay.
* **Adjust augmentation strength** carefully — too much distortion can hurt learning.
* **Explore small architectural tweaks**: fewer conv layers vs smaller fully connected layers vs adding 1×1 convolutions.

---

## 📝 Useful References

* PyTorch docs on [BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) & [Dropout](https://pytorch.org/docs/stable/nn.html#dropout)
* Tutorials showing **parameter counts calculation** in CNNs
* Resources on efficient model design for constrained environments (embedded, mobile, etc.)

---

## 📚 License / Attribution

This project is released under the MIT License.
Based on ideas from educational explorations in efficient model design.

---

## 🧮 How to Check Parameter Count (Learning Practice)

If you want to verify that your model stays under the target:

```python
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
```

Use that to guide your changes — if you enlarge something (more filters, larger FC), rerun that check.

---

Thank you for exploring — happy learning and experimenting! 🚀

```

---

If you want, I can generate a version of this README customized with exact numbers (parameter count, accuracy) pulled from your notebook, so the README matches your results. Do you want that?
::contentReference[oaicite:0]{index=0}
```

