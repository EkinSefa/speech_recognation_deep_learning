# 🎤 Speech Recognition with Deep Learning – CNN + BiGRU Approach

## 📌 Project Summary

This project aims to develop a highly accurate speech command recognition model using a *CNN + BiGRU* architecture. The *TensorFlow Speech Recognition Challenge* dataset was used to analyze 1-second WAV audio clips.

Mel-Frequency Cepstral Coefficients (MFCCs) were extracted to convert raw audio into numerical features. Then, CNN layers captured spatial features, while BiGRU layers modeled temporal patterns. The final model achieved an impressive accuracy of approximately *97%*.



## 🔍 Problem Definition

Speech command recognition systems are crucial for digital assistants, smart devices, and IoT applications. However, speech data is often affected by factors like noise, accent, and speaking speed. This project addresses these challenges by building a robust and stable speech recognition model.



## 🎯 Objective & Motivation

The goal is to develop a model that efficiently captures both spatial and temporal information using a CNN + BiGRU architecture. This hybrid approach aims to provide a reliable and scalable solution for real-world applications.



## 📚 Literature Summary

- *CNNs* are effective at capturing local patterns in audio signals.
- *GRU, especially **Bidirectional GRU (BiGRU)*, models forward and backward dependencies in time-series data.
- Combining CNNs and RNNs has shown strong performance in speech recognition tasks.



## 📁 Dataset Information

- *Name*: TensorFlow Speech Recognition Challenge  
- *Source*: [Kaggle](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data)  
- *Format*: 1-second WAV files (mono, 16 kHz)  
- *Total Samples*: ~65,000  
- *Classes*: 30 commands + unknown + background_noise



## 🧪 Preprocessing & Feature Extraction

- Raw audio was converted to MFCC features.
- Spectrogram parameters used:
  - n_fft = 384, hop_length = 160, win_length = 256
- All MFCC features were standardized using mean-std normalization.



## 🧠 CNN + BiGRU Model Architecture

### 🔹 Input Shape:
MFCC spectrograms of shape (Batch, 1, Time, Frequency)

### 🔹 CNN Layers:
python
Conv2d(1 → 32, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20))
→ BatchNorm2d → ReLU

Conv2d(32 → 32, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10))
→ BatchNorm2d → ReLU

### 🔹 BiGRU Layers:

python
BiGRU(input_size=Frequency, hidden_size=128, num_layers=2, bidirectional=True)
→ Dropout → Fully Connected → Softmax




## 📈 Results

* *Training Accuracy*: \~97%
* *Validation Accuracy*: \~96%
* *Loss Function*: Categorical Crossentropy
* *Optimizer*: Adam

### 📊 Performance Plots

*Training vs Validation Loss and Accuracy:*

![Image](https://github.com/user-attachments/assets/73817ca4-7b49-40c1-9c3c-b75f2474b09d)

**Confusion Matrix and ROC Curve:**                               
<img src="https://github.com/user-attachments/assets/fa3630b8-df8d-4d63-aa17-529e7764e58f" width="400"/>
<img src="https://github.com/user-attachments/assets/77063eae-5f5f-4d0e-91a2-ac16f225b605" width="400"/>



## 🛠️ Requirements

* Python 3.8+
* NumPy
* TensorFlow / PyTorch
* Librosa
* Scikit-learn
* Matplotlib / Seaborn

bash
pip install -r requirements.txt




## 🚀 Run the Project

bash
python main.py




## 📌 Acknowledgments

* TensorFlow Speech Recognition Challenge – Kaggle
* Inspired by hybrid CNN-RNN research in audio classification

---

## 📧 Contact

For questions or feedback, feel free to reach out at **Linkedin:** https://www.linkedin.com/in/sefa-ekin-01130a273/

