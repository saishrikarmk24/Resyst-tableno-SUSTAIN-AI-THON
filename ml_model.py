import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, Response
import threading
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from datetime import datetime
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class WaterPollutionDataset(Dataset):
    def _init_(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def _len_(self):
        return len(self.image_paths)
        
    def _getitem_(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
            image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.labels[idx]
        return image, label

class EnhancedPollutionDetector:
    def _init_(self):
        self.setup_model()
        self.setup_transforms()
        self.load_metrics_history()
        
    def setup_model(self):
        # Use EfficientNet for better performance
        self.model = models.efficientnet_b0(pretrained=True)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 5)  # Clean, Low, Medium, High, Severe
        )
        self.model.eval()
        
        # Load saved model if exists
        if os.path.exists('model_weights.pth'):
            self.model.load_state_dict(torch.load('model_weights.pth'))

    def setup_transforms(self):
        # Data augmentation for training
        self.train_transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Transform for inference
        self.inference_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load_metrics_history(self):
        self.metrics_history = []
        if os.path.exists('metrics_history.json'):
            with open('metrics_history.json', 'r') as f:
                self.metrics_history = json.load(f)

    def train_model(self, train_loader, val_loader, epochs=10):
        """Train the model with the provided data"""
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []
            
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.numpy())
                train_labels.extend(labels.numpy())
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for images, labels in val_loader:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.numpy())
                    val_labels.extend(labels.numpy())
            
            # Calculate metrics
            train_accuracy = accuracy_score(train_labels, train_preds)
            val_accuracy = accuracy_score(val_labels, val_preds)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'model_weights.pth')
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save metrics
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_loss / len(val_loader),
                'train_accuracy': float(train_accuracy),
                'val_accuracy': float(val_accuracy),
                'timestamp': datetime.now().isoformat()
            }
            self.metrics_history.append(metrics)
            
            # Save metrics history
            with open('metrics_history.json', 'w') as f:
                json.dump(self.metrics_history, f)

    def analyze_water_quality(self, image):
        """Comprehensive water quality analysis"""
        img_np = np.array(image)
        results = {}
        
        # 1. Deep Learning Analysis
        augmented = self.inference_transform(image=img_np)
        img_tensor = torch.from_numpy(augmented['image']).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
            probs = torch.softmax(predictions, dim=1)[0].numpy()
        
        pollution_classes = ['Clean', 'Low', 'Medium', 'High', 'Severe']
        predicted_class = pollution_classes[np.argmax(probs)]
        
        # 2. Advanced Color Analysis
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Detect various pollutants
        pollutants = {
            'murky_water': ([20, 50, 50], [100, 255, 255]),
            'algae': ([35, 50, 50], [85, 255, 255]),
            'oil_spill': ([0, 0, 0], [180, 255, 30]),
            'chemical_waste': ([0, 50, 50], [20, 255, 255])
        }
        
        pollutant_levels = {}
        for pollutant, (lower, upper) in pollutants.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            level = (np.sum(mask) / 255) / (mask.shape[0] * mask.shape[1]) * 100
            pollutant_levels[pollutant] = float(level)
        
        # 3. Turbidity Analysis
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        turbidity = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 4. Surface Analysis
        edges = cv2.Canny(gray, 100, 200)
        surface_disturbance = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        
        # 5. Water Clarity Analysis
        l_channel = lab[:,:,0]
        clarity_score = np.mean(l_channel) / 255 * 100
        
        # Combine all analyses
        pollution_score = (
            0.3 * pollutant_levels['murky_water'] +
            0.2 * pollutant_levels['algae'] +
            0.2 * (turbidity / 100) +
            0.15 * pollutant_levels['oil_spill'] +
            0.15 * pollutant_levels['chemical_waste']
        )
        
        results = {
            'pollution_level': float(pollution_score),
            'predicted_class': predicted_class,
            'confidence': float(np.max(probs) * 100),
            'detailed_probabilities': {
                class_name: float(prob * 100)
                for class_name, prob in zip(pollution_classes, probs)
            },
            'pollutant_levels': pollutant_levels,
            'turbidity': float(turbidity),
            'clarity_score': float(clarity_score),
            'surface_disturbance': float(surface_disturbance),
            'recommendations': self.get_recommendations(pollution_score, pollutant_levels),
            'timestamp': datetime.now().isoformat()
        }
        
        return results

    def get_recommendations(self, pollution_score, pollutant_levels):
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if pollution_score > 70:
            recommendations.append("URGENT: Water body requires immediate attention")
        
        if pollutant_levels['algae'] > 40:
            recommendations.append("High algae levels detected - Consider algae removal measures")
        
        if pollutant_levels['oil_spill'] > 20:
            recommendations.append("Possible oil contamination - Implement containment measures")
        
        if pollutant_levels['chemical_waste'] > 30:
            recommendations.append("Chemical contamination suspected - Water quality testing recommended")
        
        return recommendations

    def analyze_video_stream(self, video_source=0):
        """Real-time video analysis"""
        cap = cv2.VideoCapture(video_source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for analysis
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Analyze frame
            results = self.analyze_water_quality(pil_image)
            
            # Draw results on frame
            frame = self.draw_analysis_results(frame, results)
            
            # Convert frame to jpeg for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()

    def draw_analysis_results(self, frame, results):
        """Draw analysis results on video frame"""
        # Add text and visualizations to the frame
        cv2.putText(frame, f"Pollution Level: {results['pollution_level']:.1f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Class: {results['predicted_class']}",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

# Flask application
app = Flask(__name__)
detector = EnhancedPollutionDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        results = detector.analyze_water_quality(img)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(detector.analyze_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)