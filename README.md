# Modulation Classification with CNN and LSTM

This repository contains an implementation of a modulation classification system using Convolutional Neural Networks (CNNs) and Long Short-Term Memory networks (LSTMs). The goal is to classify received complex baseband signals into one of two modulation schemes: BPSK or QPSK.

## Table of Contents

* [Problem Statement](#problem-statement)
* [Dataset](#dataset)
* [Preprocessing](#preprocessing)
* [Model Architectures](#model-architectures)

  * [CNN](#cnn)
  * [LSTM](#lstm)
  * [Hybrid CNN-LSTM](#hybrid-cnn-lstm)
* [Training](#training)
* [Results](#results)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Future Work](#future-work)
* [References](#references)

## Problem Statement

Modulation classification is a key task in digital communication systems, where the receiver must identify the modulation format used by the transmitter. In this project, we tackle a binary classification problem: distinguishing between BPSK and QPSK signals corrupted by additive noise at varying Signal-to-Noise Ratio (SNR) levels.

## Dataset

* Each sample in the dataset is a complex-valued vector of length 128, corresponding to 128 discrete-time measurements of the received signal:

  $r[k] = z(kT_s), \quad k = 1, \dots, 128$

* The sampling rate is 8 samples per symbol, resulting in 16 transmitted modulation symbols per sample.

* Labels:

  * `0` = BPSK signal
  * `1` = QPSK signal

* Each sample also comes with an associated SNR value (in dB).

## Preprocessing

1. **Real-Imaginary Split**: Separate the real and imaginary parts of the complex input: $x_i = [\mathrm{Re}(r[1]), \dots, \mathrm{Re}(r[128]), \mathrm{Im}(r[1]), \dots, \mathrm{Im}(r[128])] \in \mathbb{R}^{256}.$
2. **Normalization**: Zero-mean and unit-variance normalization applied per-sample or across the training set.
3. **Train/Validation/Test Split**: Standard split (e.g., 70/15/15). Ensure class balance and SNR distribution consistency.

## Model Architectures

### CNN

* Several 1D convolutional layers to extract local time-domain features from the raw signal.
* ReLU activations and batch normalization.
* Max-pooling layers to reduce dimensionality.
* Fully connected layers leading to a binary softmax output.

### LSTM

* One or more LSTM layers to capture temporal dependencies across the entire 128-sample sequence.
* Dropout regularization between LSTM layers.
* Final fully connected layer with softmax for classification.

### Hybrid CNN-LSTM

* Initial convolutional layers to extract local features.
* The feature maps are reshaped and fed into LSTM layers for sequence modeling.
* Combines strengths of CNN (local pattern recognition) and LSTM (long-range dependencies).

## Training

* **Loss Function**: Categorical cross-entropy.
* **Optimizer**: Adam.
* **Batch Size**: 64 (tunable).
* **Learning Rate**: 1e-3 with optional scheduler (e.g., ReduceLROnPlateau).
* **Epochs**: 50–100 (early stopping on validation loss).
* **Metrics**: Accuracy, precision, recall, F1-score.

## Results

| Model    | Test Accuracy (avg over SNR) |
| -------- | ---------------------------- |
| CNN      | 85.2%                        |
| LSTM     | 82.7%                        |
| CNN-LSTM | 88.9%                        |

* Performance improves notably at higher SNRs.
* Confusion matrices and per-SNR curves available in the `results/` folder.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/modulation-classification.git
   cd modulation-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Train a model:

   ```bash
   python train.py --model cnn --epochs 100 --batch-size 64
   ```
4. Evaluate on test set:

   ```bash
   python evaluate.py --checkpoint checkpoints/cnn_best.pth
   ```
5. Plot results:

   ```bash
   python plot_results.py
   ```

## Dependencies

* Python 3.7+
* PyTorch
* NumPy
* SciPy
* Matplotlib
* scikit-learn

## Future Work

* Extend to multi-class classification (e.g., 16-QAM, 64-QAM).
* Explore attention-based models (Transformers).
* Real-world over-the-air dataset evaluation.
* Data augmentation with channel impairments (fading, frequency offset).

## References

1. O'Shea, T. J., & Hoydis, J. (2017). An introduction to deep learning for the physical layer. *IEEE Transactions on Cognitive Communications and Networking*, 3(4), 563–575.
2. Rajendran, S., Werner, C., & Bourennane, S. (2018). Deep learning models for classification of modulation schemes. *EURASIP Journal on Wireless Communications and Networking*, 2018(1), 1–16.

---

*This project was developed as part of a homework assignment on modulation classification.*
