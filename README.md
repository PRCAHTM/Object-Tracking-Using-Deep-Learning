# Sports Video Object Tracking System

This project was developed for the Deep Learning course under Dr. Fatemizadeh.  
The goal of the project was to design and evaluate a complete object detection and tracking pipeline for sports videos, with a focus on football matches.

The system includes:

- Object Detection using YOLOv8  
- Single Object Tracking (SOT) using CSRT  
- Multiple Object Tracking (MOT) using DeepSORT and ByteTrack  
- Quantitative evaluation using standard tracking metrics  
- Analysis of real-world challenges such as occlusion and identity switching  
- A proposed improvement using Re-Identification (Re-ID)  

---

## Project Overview

The objective was to build a full pipeline capable of:

1. Detecting players, balls, and referees in sports videos  
2. Tracking a single object across frames  
3. Tracking multiple objects while preserving identity  
4. Evaluating tracking performance using standard benchmarks  
5. Proposing and analyzing improvements  

Datasets used:

- SportsMOT (primary dataset)
- SoccerNet (additional experiments)

All experiments were conducted in Google Colab and Kaggle environments.

---

## Dataset Preparation

The SportsMOT dataset was filtered to extract football sequences only.  
Data preprocessing included:

- Extracting relevant sequences
- Converting annotations to YOLO format
- Normalizing bounding boxes
- Organizing train/validation splits
- Generating YAML configuration files

Final dataset structure:

```
yolo_dataset/
  ├── train/
  │     ├── images/
  │     └── labels/
  └── val/
        ├── images/
        └── labels/
```

This preprocessing ensured compatibility with YOLOv8 training.

---

## Object Detection – YOLOv8

### Why YOLOv8?

YOLOv8 was selected because of:

- Real-time performance
- High detection accuracy
- End-to-end training
- Strong transfer learning support

### Training Configuration

- Model: YOLOv8n (pretrained)
- Epochs: 50
- Image size: 736 × 736
- Batch size: 16
- Optimizer: AdamW
- GPU acceleration

### Detection Performance

Evaluation metrics:

- Precision: 0.9269  
- Recall: 0.9465  
- AP50: 0.9525  
- mAP: 0.7953  

The model achieved strong performance in detecting players and footballs, with only minor drops in confidence under occlusion or small object scenarios.

---

## Single Object Tracking (SOT) – CSRT

For tracking a specific object (e.g., the ball), the CSRT tracker from OpenCV was used.

### Why CSRT?

- Handles scale variation well  
- Robust to partial occlusion  
- Good balance between accuracy and speed  
- Easy integration  

### Evaluation Metrics

| Metric | Value |
|--------|--------|
| Success Score | 0.6661 |
| Precision (20px threshold) | 82.80% |
| Speed | 28.40 FPS |

### Observations

CSRT performs well when:

- Object motion is smooth
- Lighting is stable
- Occlusion is short

It struggles with:

- Long occlusion
- Rapid motion
- Severe scale change
- Strong illumination variation

A tracking heatmap was also generated to visualize movement patterns.

---

## Multiple Object Tracking (MOT)

Two algorithms were implemented and compared:

- DeepSORT  
- ByteTrack  

Both use detection outputs from YOLOv8 as input.

---

### DeepSORT

- Uses Kalman Filter for motion prediction  
- Uses deep appearance embeddings for re-identification  
- Hungarian Algorithm for assignment  

Strength:
- Higher precision

Weakness:
- More fragmentation
- Lower identity consistency

---

### ByteTrack

- Uses Kalman Filter
- IoU-based association
- Greedy matching
- Considers low-confidence detections

Strength:
- Better identity preservation
- Lower fragmentation
- Stronger MOTA

---

## Tracking Performance Comparison

### ByteTrack

- MOTA: 0.796  
- Precision: 0.9193  
- IDF1: 0.5458  
- ID Switches: 61  
- Fragmentations: 102  

### DeepSORT

- MOTA: 0.7755  
- Precision: 0.9331  
- IDF1: 0.4959  
- ID Switches: 60  
- Fragmentations: 163  

### Conclusion

ByteTrack provided better identity consistency and lower fragmentation, making it more suitable for dynamic sports scenarios.

DeepSORT achieved slightly higher precision but struggled more with identity preservation.

---

## Common Challenges in Sports Tracking

### Occlusion
Objects temporarily hidden by other players.

### ID Switch
Tracker assigns a new ID to an existing object.

### Scale Variation
Object size changes drastically due to camera distance.

### Illumination Change
Lighting shifts affecting appearance features.

These challenges directly impact IDF1 and fragmentation metrics.

---

## Proposed Improvement – Re-Identification (Re-ID)

To address occlusion, a Re-ID module was proposed:

1. Extract deep appearance features (ResNet/EfficientNet)
2. Store feature embeddings at initialization
3. Compare embeddings when object reappears
4. Combine IoU and feature distance:

```
Cost = α × IoU_cost + (1 - α) × Feature_distance
```

5. Use Hungarian algorithm for final association
6. Apply Kalman filter for motion smoothing

This hybrid motion + appearance strategy improves identity preservation during occlusion.

---

## Model Deployment and Optimization

To prepare the system for real-world deployment, several optimization techniques were analyzed:

- Quantization  
- Pruning  
- Knowledge Distillation  
- Model Compression  
- Edge Computing  

Each method balances performance, speed, and resource constraints.

Deployment ensures the model transitions from research to practical application.

---

## Technologies Used

- Python  
- PyTorch  
- YOLOv8 (Ultralytics)  
- OpenCV  
- DeepSORT  
- ByteTrack  
- NumPy  
- Google Colab / Kaggle  

---

## Course Information

Deep Learning  
Sharif University of Technology  
Dr. Fatemizadeh  

---

## Author

Parsa Hatami
