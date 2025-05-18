
# 🎥 Video Quality Evaluation – Rule-Based System

This project evaluates and compares the visual quality of two short video clips (`vid1.mp4` and `vid2.mp4`) and selects the better one based on frame-level visual cues. It is designed for podcast, interview, or conversation scenarios where multiple camera angles are available, and one must be automatically chosen.


## 📌 Objective

To develop a **rule-based logic system** (no machine learning) that analyzes the **visual quality** of each video and selects the better one. This is ideal for applications such as:

- Podcast camera switching
- Interview recordings
- Visual quality validation


## 📂 Input

- `vid1.mp4` – First 2-second video clip
- `vid2.mp4` – Second 2-second video clip


## 🧠 Features Extracted

The system evaluates each video using the following frame-level metrics:

| Metric              | Description                                                             | Preferred Value |
|---------------------|--------------------------------------------------------------------------|-----------------|
| 📏 Laplacian Sharpness     | Measures edge clarity using Laplacian variance                            | High            |
| 📐 Tenengrad Sharpness     | Measures gradient magnitude for fine details                             | High            |
| 🌀 Motion Blur Score       | Detects motion-induced blur via FFT variance                              | Low             |
| 📈 Contrast                | Difference between light and dark regions                                 | Moderate-High   |
| 💡 Brightness              | Overall light level                                                        | Moderate        |
| 🎨 Saturation              | Color richness of the scene                                                | Moderate-High   |
| 🧑‍🦱 Face Detection Score   | Area of the largest face relative to frame size                            | High            |
| ⚫ Edge Density (Optional) | Ratio of edge pixels to total pixels (indicates texture and clarity)       | High            |


## ⚙️ How It Works

1. **Frame Sampling**: Each video is split into **30 evenly spaced frames**.
2. **Metric Computation**:
   - Each frame is evaluated for sharpness, motion blur, brightness, contrast, etc.
   - Face detection is performed using Haar Cascades.
3. **Clarity Calculation**:
   - Clarity is computed by combining sharpness and motion blur penalty.
4. **Score Aggregation**:
   - All metrics are normalized and combined using weighted logic:

     Final Score = 0.4 * Clarity + 0.25 * Face Score +
                   0.15 * Contrast + 0.1 * Saturation + 0.1 * Brightness
     
5. **Best Video Selection**:
   - The video with the highest final score is selected.


## 🖥️ Output Example



vid1.mp4 → ✅ Score: 34.30 | Clarity: 85.41 | Laplacian: 5.73 | ...
vid2.mp4 → ✅ Score: 611.33 | Clarity: 1528.02 | Laplacian: 168.17 | ...
🎯 Best Video: vid2.mp4


## 🚀 Getting Started

### ✅ Requirements

- Python 3.x
- OpenCV
- NumPy

### 🔧 Installation


pip install -r requirements.txt


### ▶️ Run the Evaluation


python evaluate.py


## 📹 Optional Add-Ons

* Feature-specific output videos with overlays (sharpness, contrast, face detection)
* Edge density visualization
* Log-scaled clarity for extremely high sharpness cases



## 🧠 Why Rule-Based?

This approach is:

* Transparent and explainable
* Fast and lightweight (no training)
* Ideal for short clip evaluation and real-time scenarios


## 👩‍💻 Author

**Athira** – BVRIT | Computer Science Undergraduate
Passionate about AI, computer vision, and building practical tech solutions.



## 📃 License

MIT License. Free to use and modify with credit.

