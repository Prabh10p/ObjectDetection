import os
import io
import cv2
import base64
import numpy as np
import requests
from ultralytics import YOLO
from PIL import Image
import json

class OllamaRealityDescriber:
    def __init__(self):
        print("🚀 Loading AI Reality Describer with Ollama...")
        print("=" * 60)
        
        # Load YOLO for object detection
        print("📦 Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        
        # Target classes for better detection
        self.target_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Check Ollama availability
        print("🔍 Checking Ollama...")
        self.vision_model, self.text_model = self.check_ollama()
        
        if not self.vision_model:
            print("\n❌ Ollama not found or no vision model installed!")
            print("\n📋 Quick Setup:")
            print("   1. Install Ollama from: https://ollama.ai")
            print("   2. Run: ollama pull llava")
            print("   3. Run: ollama pull llama3.2  (for nutrition facts)")
            print("   4. Restart this script")
            print("\nPress Enter to exit...")
            input()
            exit(1)
        
        self.current_object = None
        self.current_desc = None
        self.confidence_threshold = 0.25
        self.is_loading = False
        self.all_detections = []
        self.selected_detection_idx = 0
        
        print(f"✅ Vision Model: {self.vision_model}")
        print(f"✅ Text Model: {self.text_model}")
        print("✅ AI Reality Describer Ready!")
        print("=" * 60)

    def check_ollama(self):
        """Check if Ollama is running and has necessary models."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                # Find vision model
                vision_models = ['llava', 'bakllava', 'llava-llama3', 'llava-phi3', 'moondream']
                vision_model = None
                for model_name in model_names:
                    for vm in vision_models:
                        if vm in model_name.lower():
                            vision_model = model_name
                            break
                    if vision_model:
                        break
                
                # Find text model for nutrition facts
                text_models = ['llama3.2', 'llama3.1', 'llama3', 'llama2', 'mistral', 'phi3']
                text_model = None
                for model_name in model_names:
                    for tm in text_models:
                        if tm in model_name.lower():
                            text_model = model_name
                            break
                    if text_model:
                        break
                
                if not vision_model:
                    print("\n⚠️  No vision model found!")
                    print("   Run: ollama pull llava")
                    return None, None
                
                if not text_model:
                    print("\n⚠️  No text model found for nutrition facts!")
                    print("   Run: ollama pull llama3.2")
                    print("   (Will continue with vision model only)")
                    text_model = vision_model  # Fallback to vision model
                
                return vision_model, text_model
            else:
                return None, None
        except:
            return None, None

    def get_nutrition_from_ai(self, object_name):
        """Use text-based AI to get nutrition information."""
        try:
            # Much shorter, more direct prompt
            prompt = f"""Give nutrition info for {object_name} in one sentence:

Banana example: "105 cal, potassium 422mg, vitamin B6, C, fiber 3g. Energy and heart health."
Apple example: "95 cal, vitamin C, fiber 4g. Heart health, antioxidants."

{object_name}:"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.text_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Lower for faster, more focused response
                        "num_predict": 60,   # Much shorter response
                        "num_ctx": 512,      # Smaller context window
                    }
                },
                timeout=20  # Shorter timeout
            )

            if response.status_code == 200:
                result = response.json()
                info = result.get('response', '').strip()
                
                # Clean up response
                info = info.replace('\n', ' ').strip()
                
                # Remove quotes if AI added them
                if info.startswith('"') and info.endswith('"'):
                    info = info[1:-1]
                
                # Add formatting if not present
                if info and not info.startswith('📊') and not info.startswith('ℹ️'):
                    info = f"📊 {info}"
                
                return info if info else None
            else:
                return None

        except requests.exceptions.Timeout:
            print(f"⚠️  Text AI timed out - using faster fallback")
            return None
        except Exception as e:
            print(f"⚠️  Nutrition AI error: {e}")
            return None

    def describe_with_ollama(self, frame, box, object_name):
        """Two-step AI: Vision for description + Text AI for nutrition facts."""
        try:
            # Crop the detected object
            x1, y1, x2, y2 = map(int, box)
            
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return f"Could not capture {object_name} clearly."

            # Convert to base64
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            print("   📸 Step 1: Getting visual description from AI...")
            
            # STEP 1: Vision AI describes what it sees
            vision_prompt = f"""Describe this {object_name} in 1-2 sentences. Include: color, condition, how it's positioned/held, any visible text or brands.

Be concise and specific.

Example: "A ripe yellow banana with brown spots, held up by a hand."
Example: "A bright red apple with a shiny surface on a white background."

Describe:"""

            vision_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": vision_prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 80,
                    }
                },
                timeout=60
            )

            visual_desc = ""
            if vision_response.status_code == 200:
                result = vision_response.json()
                visual_desc = result.get('response', '').strip()
                visual_desc = visual_desc.replace('\n', ' ').strip()
                
                # Clean up filler phrases
                skip_phrases = [
                    "I can see", "I notice", "In this image", "This image shows",
                    "The image depicts", "Here we have", "I observe", "Looking at",
                    "Based on the image", "From what I can see", "It appears",
                    "This appears to be", "It looks like", "This is"
                ]
                
                for phrase in skip_phrases:
                    if visual_desc.lower().startswith(phrase.lower()):
                        visual_desc = visual_desc[len(phrase):].strip()
                        if visual_desc and visual_desc[0] in ',:':
                            visual_desc = visual_desc[1:].strip()
                        if visual_desc:
                            visual_desc = visual_desc[0].upper() + visual_desc[1:]
                        break

            print("   🧠 Step 2: Getting nutrition facts from AI...")
            
            # STEP 2: Text AI provides nutrition/facts
            nutrition_info = self.get_nutrition_from_ai(object_name)

            # Combine both
            final_response = ""
            if visual_desc:
                final_response = visual_desc
            
            if nutrition_info:
                if final_response:
                    final_response += " " + nutrition_info
                else:
                    final_response = nutrition_info
            
            if not final_response:
                final_response = f"Detected: {object_name}"
            
            return final_response

        except Exception as e:
            print(f"❌ AI Error: {e}")
            return f"Error analyzing {object_name}"

    def detect_objects(self, frame):
        """Detect objects in frame using YOLO."""
        results = self.model(frame, verbose=False, conf=0.25)
        detections = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > self.confidence_threshold:
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    
                    if name in self.target_classes:
                        bbox = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'name': name,
                            'confidence': conf,
                            'box': bbox
                        })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[:10]

    def draw_boxes(self, frame):
        """Draw bounding boxes around detected objects."""
        for i, det in enumerate(self.all_detections):
            x1, y1, x2, y2 = map(int, det['box'])
            
            color = (0, 255, 255) if i == self.selected_detection_idx else (0, 255, 0)
            thickness = 3 if i == self.selected_detection_idx else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{det['name']} {det['confidence']:.0%}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def draw_description_box(self, frame, text):
        """Draw description box at bottom of screen."""
        height, width = frame.shape[:2]
        box_height = 200
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - box_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        cv2.rectangle(frame, (0, height - box_height), (width, height), (0, 255, 255), 2)
        
        cv2.putText(frame, "🤖 AI Analysis:", (20, height - box_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Wrap text
        words = text.split()
        lines = []
        current_line = []
        max_width = width - 40
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
            
            if text_size < max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        y_offset = height - box_height + 55
        for line in lines[:6]:
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 23

    def draw_ui(self, frame):
        """Draw UI elements."""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        instructions = "SPACE: AI Analysis + Nutrition | UP/DOWN: Select | Q: Quit"
        cv2.putText(frame, instructions, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Main loop for camera feed and object detection."""
        video = cv2.VideoCapture(0)
        
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video.set(cv2.CAP_PROP_FPS, 30)
        video.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        if not video.isOpened():
            print("❌ Error: Could not open camera!")
            return

        print("\n🎮 Controls:")
        print("   [SPACE]    ➜ Get AI description + nutrition facts")
        print("   [UP/DOWN]  ➜ Select different objects")
        print("   [Q]        ➜ Quit")
        print("\n🤖 This uses TWO AI models:")
        print("   1. Vision AI (LLaVA) - Describes what it sees")
        print("   2. Text AI (Llama) - Provides nutrition facts")
        print("\n🎥 Camera ready! Point at objects and press SPACE\n")

        while True:
            ret, frame = video.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            self.all_detections = self.detect_objects(frame)
            self.draw_ui(frame)

            if self.all_detections:
                self.selected_detection_idx = min(
                    self.selected_detection_idx, 
                    len(self.all_detections) - 1
                )
                
                self.draw_boxes(frame)

                det = self.all_detections[self.selected_detection_idx]
                if self.current_object == det['name'] and self.current_desc:
                    self.draw_description_box(frame, self.current_desc)
                else:
                    hint = f"Selected: {det['name']} - Press SPACE for AI analysis!"
                    cv2.putText(frame, hint, (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "No objects detected - Point camera at objects", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if self.is_loading:
                height = frame.shape[0]
                cv2.putText(frame, "🤖 AI is analyzing (2-step process)...", (20, height - 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.imshow("AI Nutrition Analyzer", frame)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\n👋 Goodbye!")
                break
                
            elif key == ord(' '):
                if self.all_detections and not self.is_loading:
                    det = self.all_detections[self.selected_detection_idx]
                    self.is_loading = True
                    
                    print(f"\n🔍 Analyzing {det['name']}...")
                    
                    desc = self.describe_with_ollama(frame, det['box'], det['name'])
                    
                    self.current_object = det['name']
                    self.current_desc = desc
                    self.is_loading = False
                    
                    print(f"✅ {desc}\n")
                    
            elif key == 82 or key == 0:  # UP
                if self.all_detections:
                    self.selected_detection_idx = (self.selected_detection_idx - 1) % len(self.all_detections)
                    self.current_desc = None
                    
            elif key == 84 or key == 1:  # DOWN
                if self.all_detections:
                    self.selected_detection_idx = (self.selected_detection_idx + 1) % len(self.all_detections)
                    self.current_desc = None

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌟 DUAL-AI NUTRITION ANALYZER 🌟")
    print("=" * 60)
    print("\n💡 Uses TWO AI Models for Best Results:")
    print("   🎨 Vision AI - Describes appearance")
    print("   🧠 Text AI - Provides nutrition facts")
    print("\n📋 Setup Required:")
    print("   1. ollama pull llava      (for vision)")
    print("   2. ollama pull llama3.2   (for nutrition)")
    print("\n" + "=" * 60 + "\n")
    
    try:
        describer = OllamaRealityDescriber()
        describer.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")