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
        
        # Target classes for better detection (COCO dataset class names)
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
        self.ollama_model = self.check_ollama()
        
        if not self.ollama_model:
            print("\n❌ Ollama not found or no vision model installed!")
            print("\n📋 Quick Setup:")
            print("   1. Install Ollama from: https://ollama.ai")
            print("   2. Run: ollama pull llava")
            print("   3. Restart this script")
            print("\nPress Enter to exit...")
            input()
            exit(1)
        
        self.current_object = None
        self.current_desc = None
        self.confidence_threshold = 0.25  # Lowered for better detection
        self.is_loading = False
        self.all_detections = []
        self.selected_detection_idx = 0
        
        print(f"✅ Using Ollama model: {self.ollama_model}")
        print("✅ AI Reality Describer Ready!")
        print("=" * 60)

    def check_ollama(self):
        """Check if Ollama is running and has a vision model."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                
                # List of vision-capable models
                vision_models = ['llava', 'bakllava', 'llava-llama3', 'llava-phi3', 'moondream']
                
                for model in models:
                    model_name = model.get('name', '')
                    for vm in vision_models:
                        if vm in model_name.lower():
                            return model_name
                
                print("\n⚠️  Ollama is running but no vision model found.")
                print("   Available models:", [m.get('name') for m in models])
                print("   Please run: ollama pull llava")
                return None
            else:
                print("\n⚠️  Ollama responded with error:", response.status_code)
                return None
        except requests.exceptions.ConnectionError:
            print("\n⚠️  Cannot connect to Ollama. Is it running?")
            print("   Start Ollama or install from: https://ollama.ai")
            return None
        except Exception as e:
            print(f"\n⚠️  Error checking Ollama: {e}")
            return None

    def describe_with_ollama(self, frame, box, object_name):
        """Use Ollama's vision model to describe the detected object."""
        try:
            # Crop the detected object with some padding for context
            x1, y1, x2, y2 = map(int, box)
            
            # Add padding (10% on each side)
            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return "Could not capture object clearly."

            # Convert to base64 JPEG
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Enhanced prompt with specific instructions including facts/nutrition
            prompt = f"""You are a detailed visual narrator and information provider. The object detection system identified this as a '{object_name}'.

PART 1 - VISUAL DESCRIPTION (2-3 sentences):
Describe what you see:
- APPEARANCE: Colors, patterns, materials, textures, condition
- SPECIFIC DETAILS: Text, logos, distinctive features
- CONTEXT: What's happening, how it's being used, setting

For PEOPLE: Describe clothing, actions (holding phone, talking, etc.), posture, accessories

PART 2 - FACTS & INFORMATION (2-3 sentences):
Provide relevant facts based on what you identified:

For FOOD items (fruits, vegetables, snacks, meals):
- Nutritional info: Approximate calories per serving (e.g., "A medium orange has about 60-80 calories")
- Key vitamins/nutrients (e.g., "Rich in Vitamin C, fiber, and antioxidants")
- Health benefits (e.g., "Good for immune system and skin health")

For ELECTRONICS (phone, laptop, etc.):
- Common uses and features
- Typical functions or capabilities

For DRINKS (bottle, cup, coffee, etc.):
- What it likely contains
- Nutritional aspects if identifiable

For OTHER OBJECTS:
- Primary purpose and common uses
- Interesting facts or context

FORMATTING RULES:
- Write in natural, conversational sentences
- Start visual description immediately (no preamble like "I see" or "This shows")
- Separate description and facts naturally
- Be specific and engaging
- No bullet points in the response

EXAMPLE for an orange:
"An orange citrus fruit with a bright, textured peel is being held up, displaying its vibrant orange color against the background. The fruit appears fresh and ripe. Oranges are an excellent source of Vitamin C, with a medium-sized orange containing about 70 calories and providing over 100% of the daily recommended Vitamin C intake. They're also rich in fiber, potassium, and antioxidants that support immune function and heart health."

EXAMPLE for a cell phone:
"A person wearing a white shirt is holding a smartphone up to their ear, appearing to be engaged in a phone call. The device looks like a modern smartphone with a dark-colored case. Smartphones serve as multi-functional devices combining communication, internet access, photography, and countless apps. They typically feature touchscreens, high-resolution cameras, and powerful processors that enable everything from video calls to mobile banking."

Now describe what you see and provide relevant facts:"""

            # Call Ollama API with adjusted parameters
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,  # Balanced for descriptions and facts
                        "top_p": 0.95,
                        "num_predict": 200,  # Longer for facts + description
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                description = result.get('response', '').strip()
                
                # Clean up the description
                description = description.replace('\n', ' ').strip()
                
                # Remove any meta-commentary
                skip_phrases = [
                    "I can see", "I notice", "In this image", "This image shows",
                    "The image depicts", "Here we have", "I observe"
                ]
                for phrase in skip_phrases:
                    if description.lower().startswith(phrase.lower()):
                        description = description[len(phrase):].strip()
                        if description.startswith(','):
                            description = description[1:].strip()
                        # Capitalize first letter
                        if description:
                            description = description[0].upper() + description[1:]
                
                if description:
                    return description
                else:
                    return f"A {object_name} is visible in the frame."
            else:
                print(f"Ollama API error: {response.status_code}")
                return f"Detected as {object_name}. Description failed."

        except requests.exceptions.Timeout:
            print("⚠️  Ollama took too long to respond (timeout)")
            return f"Detected as {object_name}. AI timed out."
        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            return f"Detected as {object_name}. AI error occurred."

    def detect_objects(self, frame):
        """Detect objects in frame using YOLO - focusing on common everyday items."""
        results = self.model(frame, verbose=False, conf=0.25)  # Lower confidence for more detections
        detections = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > self.confidence_threshold:
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    
                    # Only include objects from our target classes
                    if name in self.target_classes:
                        bbox = box.xyxy[0].cpu().numpy()
                        detections.append({
                            'name': name,
                            'confidence': conf,
                            'box': bbox
                        })
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Limit to top 10 detections to avoid clutter
        return detections[:10]

    def draw_boxes(self, frame):
        """Draw bounding boxes around detected objects."""
        for i, det in enumerate(self.all_detections):
            x1, y1, x2, y2 = map(int, det['box'])
            
            # Yellow for selected, green for others
            color = (0, 255, 255) if i == self.selected_detection_idx else (0, 255, 0)
            thickness = 3 if i == self.selected_detection_idx else 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Label with object name and confidence
            label = f"{det['name']} {det['confidence']:.0%}"
            
            # Background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def draw_description_box(self, frame, text):
        """Draw a semi-transparent box with AI description at the bottom."""
        height, width = frame.shape[:2]
        box_height = 200  # Increased height for description + facts
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - box_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Add border
        cv2.rectangle(frame, (0, height - box_height), (width, height), (0, 255, 255), 2)
        
        # Add title with icon
        cv2.putText(frame, "AI Analysis:", (20, height - box_height + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Wrap text to fit width
        words = text.split()
        lines = []
        current_line = []
        max_width = width - 40
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0]
            
            if text_size < max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Draw text lines
        y_offset = height - box_height + 55
        for line in lines[:6]:  # Max 6 lines for description + facts
            cv2.putText(frame, line, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            y_offset += 25

    def draw_ui(self, frame):
        """Draw UI elements (instructions, status)."""
        height, width = frame.shape[:2]
        
        # Top instruction bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        instructions = "SPACE: Describe | UP/DOWN: Select | Q: Quit"
        cv2.putText(frame, instructions, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Main loop for camera feed and object detection."""
        video = cv2.VideoCapture(0)
        
        # Set camera resolution and optimize for detection
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video.set(cv2.CAP_PROP_FPS, 30)
        video.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        
        if not video.isOpened():
            print("❌ Error: Could not open camera!")
            print("   Trying alternative camera indices...")
            for i in range(1, 5):
                video = cv2.VideoCapture(i)
                if video.isOpened():
                    print(f"✅ Found camera at index {i}")
                    break
            if not video.isOpened():
                print("❌ No camera found!")
                return

        print("\n🎮 Controls:")
        print("   [SPACE]    ➜ Get AI description of selected object")
        print("   [UP/DOWN]  ➜ Cycle through detected objects")
        print("   [Q]        ➜ Quit")
        print("\n🎥 Camera started! Point at objects and press SPACE")
        print("💡 TIPS:")
        print("   • First description may take 30-60 seconds (model loading)")
        print("   • Hold objects closer to camera for better detection")
        print("   • Good lighting helps detection accuracy")
        print("   • Detects: phones, glasses, food, bottles, books, etc.\n")

        frame_skip = 0
        while True:
            ret, frame = video.read()
            if not ret:
                print("⚠️  Lost camera connection, attempting to reconnect...")
                video.release()
                video = cv2.VideoCapture(0)
                continue
            
            # Flip frame horizontally (mirror effect)
            frame = cv2.flip(frame, 1)
            
            # Skip frames for better performance when loading
            if self.is_loading:
                frame_skip += 1
                if frame_skip % 5 != 0:
                    cv2.imshow("Ollama AI Reality Describer", frame)
                    cv2.waitKey(1)
                    continue
            
            # Detect objects in current frame
            self.all_detections = self.detect_objects(frame)

            # Draw UI
            self.draw_ui(frame)

            if self.all_detections:
                # Ensure selected index is valid
                self.selected_detection_idx = min(
                    self.selected_detection_idx, 
                    len(self.all_detections) - 1
                )
                
                # Draw bounding boxes
                self.draw_boxes(frame)

                # Show description if available
                det = self.all_detections[self.selected_detection_idx]
                if self.current_object == det['name'] and self.current_desc:
                    self.draw_description_box(frame, self.current_desc)
                else:
                    # Show hint
                    hint = f"Selected: {det['name']} - Press SPACE for AI analysis + facts"
                    cv2.putText(frame, hint, (20, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # No objects detected
                cv2.putText(frame, "No objects detected - Move camera around", 
                           (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Show loading indicator
            if self.is_loading:
                height = frame.shape[0]
                cv2.putText(frame, "🤖 AI is analyzing (description + facts)...", (20, height - 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Display frame
            cv2.imshow("Ollama AI Reality Describer", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                print("\n👋 Goodbye!")
                break
                
            elif key == ord(' '):
                # SPACE pressed - get description
                if self.all_detections and not self.is_loading:
                    det = self.all_detections[self.selected_detection_idx]
                    self.is_loading = True
                    
                    print(f"\n🔍 Analyzing {det['name']} (description + facts)...")
                    print("   ⏳ First time may take 30-60 seconds (loading model)")
                    print("   ⚡ Next times will be much faster (5-15 seconds)")
                    print("   📊 Getting visual description + nutritional/factual info")
                    print("   ℹ️  The camera feed may freeze briefly - this is normal!")
                    
                    # Get description from Ollama
                    desc = self.describe_with_ollama(frame, det['box'], det['name'])
                    
                    self.current_object = det['name']
                    self.current_desc = desc
                    self.is_loading = False
                    
                    print(f"✅ Description: {desc}\n")
                    
            elif key == 82 or key == 0:  # UP arrow
                if self.all_detections:
                    self.selected_detection_idx = (self.selected_detection_idx - 1) % len(self.all_detections)
                    self.current_desc = None
                    
            elif key == 84 or key == 1:  # DOWN arrow
                if self.all_detections:
                    self.selected_detection_idx = (self.selected_detection_idx + 1) % len(self.all_detections)
                    self.current_desc = None

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🌟 AI REALITY DESCRIBER - POWERED BY OLLAMA 🌟")
    print("=" * 60)
    print("\n💡 This app uses 100% FREE local AI (Ollama)")
    print("   No API keys, no internet, no limits!\n")
    
    # Check if this is first run
    print("📋 First Time Setup:")
    print("   1. Install Ollama: https://ollama.ai")
    print("   2. Run: ollama pull llava")
    print("   3. Run this script again")
    print("\n" + "=" * 60 + "\n")
    
    try:
        describer = OllamaRealityDescriber()
        describer.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nIf you see connection errors, make sure:")
        print("   1. Ollama is installed and running")
        print("   2. You've downloaded a vision model: ollama pull llava")