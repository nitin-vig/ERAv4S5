# MNIST Training with <20,000 Parameters

This repository demonstrates training a compact convolutional neural network on the MNIST dataset with **fewer than 20,000 trainable parameters**, while still achieving high accuracy. Emphasis is placed on understanding design choices, trade‑offs, and techniques for efficient learning.

---

## 📂 Repository Contents

- `MNIST_Training_in_less_than_20k_Params.ipynb` — Jupyter notebook showing the full process: model definition, training, validation, and evaluation.  
- `data/` — MNIST data is downloaded here via torchvision.  
- `README.md` — This file.  

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
| **Dropout (light, after FC layer, after activation)** | Helps generalization without crippling the model | Dropout after ReLU in FC removes random active units, reducing co‑adaptation | Avoided in small conv layers.
| **Data Augmentation** | Small rotations / translations (via `RandomRotation`, `RandomAffine`) | Makes model robust to small input perturbations; helps reduce overfit |
| **Optimizer & Learning Rate** | SGD with momentum, moderate LR (e.g. 0.01), with scheduling | Momentum helps smooth optimization; scheduling avoids getting stuck at accuracy plateaus | Idea is to start with higher LR and reduce it over time to fine tune.
| **Training / Validation Tracking** | Monitor both loss & accuracy over epochs | Ensures you see overfitting vs under‑fitting; decide whether to increase capacity or regularization |

---

## ⚙ How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/nitin-vig/ERAv4S5.git
   cd ERAv4S5
   ````

2. Install dependencies

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

* **Try different batch sizes. Batch size and Learning can usually be adjusted to same multiplier to speed up learning. 
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



If you want, I can generate a version of this README customized with exact numbers (parameter count, accuracy) pulled from your notebook, so the README matches your results. Do you want that?
::contentReference[oaicite:0]{index=0}
```

Logs:
 Epoch: 1 : 
loss=0.2783926725387573 batch_id=468: 100%|██████████| 469/469 [00:56<00:00,  8.36it/s]
Test set: Average loss: 0.0851, Accuracy: 9724/10000 (97.24%)


 Epoch: 2 : 
loss=0.16270093619823456 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.19it/s]
Test set: Average loss: 0.0417, Accuracy: 9864/10000 (98.64%)


 Epoch: 3 : 
loss=0.3241729140281677 batch_id=468: 100%|██████████| 469/469 [00:49<00:00,  9.41it/s]
Test set: Average loss: 0.0450, Accuracy: 9849/10000 (98.49%)


 Epoch: 4 : 
loss=0.16042646765708923 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.33it/s]
Test set: Average loss: 0.0320, Accuracy: 9913/10000 (99.13%)


 Epoch: 5 : 
loss=0.09749811887741089 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.33it/s]
Test set: Average loss: 0.0357, Accuracy: 9882/10000 (98.82%)


 Epoch: 6 : 
loss=0.2068089246749878 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.06it/s]
Test set: Average loss: 0.0252, Accuracy: 9922/10000 (99.22%)


 Epoch: 7 : 
loss=0.07782670110464096 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.27it/s]
Test set: Average loss: 0.0229, Accuracy: 9924/10000 (99.24%)


 Epoch: 8 : 
loss=0.050765182822942734 batch_id=468: 100%|██████████| 469/469 [00:52<00:00,  8.97it/s]
Test set: Average loss: 0.0239, Accuracy: 9930/10000 (99.30%)


 Epoch: 9 : 
loss=0.09250655770301819 batch_id=468: 100%|██████████| 469/469 [00:58<00:00,  7.98it/s]
Test set: Average loss: 0.0236, Accuracy: 9923/10000 (99.23%)


 Epoch: 10 : 
loss=0.13224434852600098 batch_id=468: 100%|██████████| 469/469 [00:52<00:00,  8.88it/s]
Test set: Average loss: 0.0204, Accuracy: 9932/10000 (99.32%)


 Epoch: 11 : 
loss=0.0859462320804596 batch_id=468: 100%|██████████| 469/469 [00:52<00:00,  8.87it/s]
Test set: Average loss: 0.0216, Accuracy: 9930/10000 (99.30%)


 Epoch: 12 : 
loss=0.1554780751466751 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.11it/s]
Test set: Average loss: 0.0211, Accuracy: 9936/10000 (99.36%)


 Epoch: 13 : 
loss=0.14891232550144196 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.19it/s]
Test set: Average loss: 0.0206, Accuracy: 9938/10000 (99.38%)


 Epoch: 14 : 
loss=0.03960130736231804 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.26it/s]
Test set: Average loss: 0.0201, Accuracy: 9932/10000 (99.32%)


 Epoch: 15 : 
loss=0.09557536989450455 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.35it/s]
Test set: Average loss: 0.0223, Accuracy: 9934/10000 (99.34%)


 Epoch: 16 : 
loss=0.05995398387312889 batch_id=468: 100%|██████████| 469/469 [00:50<00:00,  9.31it/s]
Test set: Average loss: 0.0201, Accuracy: 9943/10000 (99.43%)


 Epoch: 17 : 
loss=0.05822938680648804 batch_id=468: 100%|██████████| 469/469 [00:54<00:00,  8.53it/s]
Test set: Average loss: 0.0198, Accuracy: 9936/10000 (99.36%)


 Epoch: 18 : 
loss=0.09200435876846313 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.11it/s]
Test set: Average loss: 0.0190, Accuracy: 9941/10000 (99.41%)


 Epoch: 19 : 
loss=0.07043924182653427 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.02it/s]
Test set: Average loss: 0.0205, Accuracy: 9937/10000 (99.37%)


 Epoch: 20 : 
loss=0.06360998004674911 batch_id=468: 100%|██████████| 469/469 [00:51<00:00,  9.14it/s]
Test set: Average loss: 0.0187, Accuracy: 9940/10000 (99.40%)
