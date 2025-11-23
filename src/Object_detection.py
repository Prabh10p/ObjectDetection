import os
import io
import cv2
import base64
import numpy as np
import requests
from ultralytics import YOLO
from PIL import Image
import json

class UniversalSmartAnalyzer:
    def __init__(self):
        print("🚀 Loading Universal Smart Analyzer...")
        print("=" * 60)
        
        # Load YOLO for initial detection
        print("📦 Loading YOLO detection model...")
        self.model = YOLO('yolov8n.pt')
        
        # Check Ollama availability
        print("🔍 Checking Ollama AI...")
        self.vision_model, self.text_model = self.check_ollama()
        
        if not self.vision_model:
            print("\n❌ Ollama not found or no vision model installed!")
            print("\n📋 Quick Setup:")
            print("   1. Install Ollama from: https://ollama.ai")
            print("   2. Run: ollama pull llava")
            print("   3. Run: ollama pull llama3.2")
            print("   4. Restart this script")
            print("\nPress Enter to exit...")
            input()
            exit(1)
        
        self.current_desc = None
        self.confidence_threshold = 0.20
        self.is_loading = False
        self.all_detections = []
        self.selected_detection_idx = 0
        self.analysis_mode = "auto"  # auto, full_frame, or selected
        
        print(f"✅ Vision Model: {self.vision_model}")
        print(f"✅ Text Model: {self.text_model}")
        print("✅ Universal Smart Analyzer Ready!")
        print("=" * 60)

    def check_ollama(self):
        """Check if Ollama is running and has necessary models."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                # Prioritize better vision models (llava is best, moondream is basic)
                vision_models = ['llava', 'bakllava', 'llava-llama3', 'llava-phi3', 'moondream']
                vision_model = None
                for model_name in model_names:
                    for vm in vision_models:
                        if vm in model_name.lower():
                            vision_model = model_name
                            break
                    if vision_model:
                        break
                
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
                    return None, None
                if not text_model:
                    text_model = vision_model
                
                # Warn if using moondream (it gives short responses)
                if 'moondream' in vision_model.lower():
                    print("\n⚠️  WARNING: Using moondream (gives short responses)")
                    print("💡 For MUCH better results, install llava:")
                    print("   Run: ollama pull llava")
                    print("   Then restart this program!\n")
                
                return vision_model, text_model
            else:
                return None, None
        except:
            return None, None

    def analyze_with_ai(self, frame, box=None, detected_class=None):
        """
        Universal AI analyzer - identifies and describes anything in the image.
        Can analyze full frame or specific detected object.
        """
        try:
            # Prepare image (crop if box provided, otherwise use full frame)
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    return "Could not capture image clearly."
                analysis_img = crop
            else:
                analysis_img = frame

            # Convert to base64
            pil_img = Image.fromarray(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            print("   🧠 AI Vision Analysis in progress...")
            
            # Check if using moondream (needs multiple questions)
            is_moondream = 'moondream' in self.vision_model.lower()
            
            if is_moondream:
                # Moondream works better with multiple simple questions
                return self.analyze_with_moondream(image_base64, detected_class)
            
            # For llava and other models - use detailed prompts
            context_hint = f"YOLO detected this as '{detected_class}'. " if detected_class else ""
            
            # Create specific prompt based on what was detected
            if detected_class and detected_class == 'person':
                vision_prompt = f"""Describe this person in detail. Include:
- Gender and approximate age range
- Physical appearance (hair, facial features, skin tone)
- Clothing (colors, style, type of clothes)
- Accessories (glasses, jewelry, watch, etc.)
- Facial expression and body language
- Overall impression and context

Write 4-5 detailed sentences. Be specific and descriptive."""

            elif detected_class and detected_class in ['cell phone', 'laptop', 'mouse', 'keyboard', 'tv', 'remote']:
                vision_prompt = f"""This is a {detected_class}. Analyze it in detail:
- EXACT brand name (look for logos, text, design)
- Specific model if visible
- Color and materials
- Condition (new, used, scratched, etc.)
- Distinctive features or buttons
- Any visible text or markings

Write 4-5 detailed sentences. Be VERY specific about brand and model."""

            elif detected_class and detected_class in ['car', 'truck', 'motorcycle', 'bicycle']:
                vision_prompt = f"""This is a {detected_class}. Describe it:
- Make and model if identifiable
- Color and body style
- Year or generation (if visible)
- Condition and any modifications
- Distinctive features

Write 4-5 detailed sentences."""

            elif detected_class and detected_class in ['cat', 'dog', 'bird', 'horse']:
                vision_prompt = f"""This is a {detected_class}. Describe it:
- Breed or type
- Color and markings
- Size and build
- Age (puppy/kitten, adult, senior)
- Grooming and condition
- Personality visible in expression

Write 4-5 detailed sentences."""

            elif detected_class and detected_class in ['bottle', 'cup', 'wine glass', 'bowl']:
                vision_prompt = f"""This is a {detected_class}. Describe:
- What's in it (drink, food, empty?)
- Brand if visible
- Material (glass, plastic, metal, ceramic)
- Size and shape
- Design and color
- Condition

Write 3-4 sentences."""

            else:
                vision_prompt = f"""{context_hint}Look at this image very carefully and describe EVERYTHING you see in great detail.

For PRODUCTS (phone, laptop, mouse, headphones, etc.):
- BRAND NAME (look for logos!)
- Model number or name
- Color and materials
- Condition and features
- Any text visible

For PEOPLE:
- Age range and gender
- Physical features
- Clothing and accessories
- Expression and pose

For ANIMALS:
- Species and breed
- Color and markings
- Size and features

For ANYTHING ELSE:
- What exactly is it?
- Colors, materials, condition
- Size and distinctive features
- Any text or brands visible

Write AT LEAST 4-5 detailed sentences. Be VERY specific!"""

            vision_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": vision_prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 250,
                        "top_p": 0.9,
                    }
                },
                timeout=120
            )

            if vision_response.status_code != 200:
                return f"❌ AI analysis failed (HTTP {vision_response.status_code}). Ollama may be busy."

            result = vision_response.json()
            visual_analysis = result.get('response', '').strip()
            
            print(f"   📝 Raw AI response length: {len(visual_analysis)} characters")
            
            if not visual_analysis or len(visual_analysis) < 20:
                return f"⚠️ AI gave very short response. Try again or install llava:\n  ollama pull llava\n\nRaw: {visual_analysis}"

            # Clean up response
            skip_phrases = [
                "I can see", "I notice", "In this image", "This image shows",
                "The image depicts", "Here we have", "I observe", "Looking at",
                "Based on the image", "From what I can see", "It appears that",
                "This appears to be", "It looks like", "I see", "Based on what I can see",
                "In the image", "The image contains", "Here I see"
            ]
            
            original_length = len(visual_analysis)
            for phrase in skip_phrases:
                if visual_analysis.lower().startswith(phrase.lower()):
                    visual_analysis = visual_analysis[len(phrase):].strip()
                    if visual_analysis and visual_analysis[0] in ',:':
                        visual_analysis = visual_analysis[1:].strip()
                    if visual_analysis:
                        visual_analysis = visual_analysis[0].upper() + visual_analysis[1:]
                    break
            
            if len(visual_analysis) < original_length * 0.3:
                visual_analysis = result.get('response', '').strip()

            # Step 2: Get additional context/info if it's a product
            if self.is_likely_product(visual_analysis) or (detected_class and detected_class in ['cell phone', 'laptop', 'mouse', 'keyboard']):
                print("   💡 Fetching product details and pricing...")
                additional_info = self.get_product_context(visual_analysis)
                if additional_info:
                    return f"🔍 Visual Analysis:\n{visual_analysis}\n\n💰 Product Info:\n{additional_info}"
            
            return f"🔍 Analysis:\n{visual_analysis}"

        except Exception as e:
            print(f"❌ AI Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during analysis: {str(e)}\n\nCheck that Ollama is running: ollama list"
    
    def analyze_with_moondream(self, image_base64, detected_class):
        """Special handler for moondream - asks multiple simple questions."""
        print("   🌙 Using moondream multi-question mode...")
        
        questions = []
        
        if detected_class == 'person':
            questions = [
                "What does this person look like? Describe their appearance.",
                "What is this person wearing?",
                "What is this person's approximate age and gender?"
            ]
        elif detected_class in ['cell phone', 'laptop', 'mouse', 'keyboard', 'tv']:
            questions = [
                "What brand and model is this device? Look for logos.",
                "What color is this device?",
                "What condition is this device in? Any visible features?"
            ]
        elif detected_class in ['cat', 'dog', 'bird']:
            questions = [
                "What type and breed is this animal?",
                "What color is this animal? Any markings?",
                "Describe this animal's appearance."
            ]
        else:
            questions = [
                "What exactly is in this image?",
                "Describe its color, size, and appearance.",
                "Are there any brands, text, or distinctive features visible?"
            ]
        
        answers = []
        for i, question in enumerate(questions, 1):
            print(f"   ❓ Question {i}/{len(questions)}: {question[:50]}...")
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.vision_model,
                        "prompt": question,
                        "images": [image_base64],
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 100}
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                    if answer and len(answer) > 5:
                        answers.append(answer)
            except:
                pass
        
        if answers:
            combined = " ".join(answers)
            return f"🔍 Analysis:\n{combined}"
        else:
            return "⚠️ Could not analyze. Please install llava:\n  ollama pull llava"
        """
        Universal AI analyzer - identifies and describes anything in the image.
        Can analyze full frame or specific detected object.
        """
        try:
            # Prepare image (crop if box provided, otherwise use full frame)
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                pad_x = int((x2 - x1) * 0.15)
                pad_y = int((y2 - y1) * 0.15)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    return "Could not capture image clearly."
                analysis_img = crop
            else:
                analysis_img = frame

            # Convert to base64
            pil_img = Image.fromarray(cv2.cvtColor(analysis_img, cv2.COLOR_BGR2RGB))
            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=95)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            print("   🧠 AI Vision Analysis in progress...")
            
            # Step 1: Visual identification with targeted prompts based on detected class
            context_hint = f"YOLO detected this as '{detected_class}'. " if detected_class else ""
            
            # Create specific prompt based on what was detected
            if detected_class and detected_class == 'person':
                vision_prompt = f"""Describe this person in detail. Include:
- Gender and approximate age range
- Physical appearance (hair, facial features, skin tone)
- Clothing (colors, style, type of clothes)
- Accessories (glasses, jewelry, watch, etc.)
- Facial expression and body language
- Overall impression and context

Write 4-5 detailed sentences. Be specific and descriptive."""

            elif detected_class and detected_class in ['cell phone', 'laptop', 'mouse', 'keyboard', 'tv', 'remote']:
                vision_prompt = f"""This is a {detected_class}. Analyze it in detail:
- EXACT brand name (look for logos, text, design)
- Specific model if visible
- Color and materials
- Condition (new, used, scratched, etc.)
- Distinctive features or buttons
- Any visible text or markings

Write 4-5 detailed sentences. Be VERY specific about brand and model."""

            elif detected_class and detected_class in ['car', 'truck', 'motorcycle', 'bicycle']:
                vision_prompt = f"""This is a {detected_class}. Describe it:
- Make and model if identifiable
- Color and body style
- Year or generation (if visible)
- Condition and any modifications
- Distinctive features

Write 4-5 detailed sentences."""

            elif detected_class and detected_class in ['cat', 'dog', 'bird', 'horse']:
                vision_prompt = f"""This is a {detected_class}. Describe it:
- Breed or type
- Color and markings
- Size and build
- Age (puppy/kitten, adult, senior)
- Grooming and condition
- Personality visible in expression

Write 4-5 detailed sentences."""

            elif detected_class and detected_class in ['bottle', 'cup', 'wine glass', 'bowl']:
                vision_prompt = f"""This is a {detected_class}. Describe:
- What's in it (drink, food, empty?)
- Brand if visible
- Material (glass, plastic, metal, ceramic)
- Size and shape
- Design and color
- Condition

Write 3-4 sentences."""

            else:
                vision_prompt = f"""{context_hint}Look at this image very carefully and describe EVERYTHING you see in great detail.

For PRODUCTS (phone, laptop, mouse, headphones, etc.):
- BRAND NAME (look for logos!)
- Model number or name
- Color and materials
- Condition and features
- Any text visible

For PEOPLE:
- Age range and gender
- Physical features
- Clothing and accessories
- Expression and pose

For ANIMALS:
- Species and breed
- Color and markings
- Size and features

For ANYTHING ELSE:
- What exactly is it?
- Colors, materials, condition
- Size and distinctive features
- Any text or brands visible

Write AT LEAST 4-5 detailed sentences. Be VERY specific!"""

            vision_response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.vision_model,
                    "prompt": vision_prompt,
                    "images": [image_base64],
                    "stream": False,
                    "options": {
                        "temperature": 0.4,
                        "num_predict": 250,
                        "top_p": 0.9,
                    }
                },
                timeout=120
            )

            if vision_response.status_code != 200:
                return f"❌ AI analysis failed (HTTP {vision_response.status_code}). Ollama may be busy."

            result = vision_response.json()
            visual_analysis = result.get('response', '').strip()
            
            print(f"   📝 Raw AI response length: {len(visual_analysis)} characters")
            
            if not visual_analysis or len(visual_analysis) < 20:
                return f"⚠️ AI gave very short response. Try again or check Ollama.\n\nRaw: {visual_analysis}"

            # Clean up response
            skip_phrases = [
                "I can see", "I notice", "In this image", "This image shows",
                "The image depicts", "Here we have", "I observe", "Looking at",
                "Based on the image", "From what I can see", "It appears that",
                "This appears to be", "It looks like", "I see", "Based on what I can see",
                "In the image", "The image contains", "Here I see"
            ]
            
            original_length = len(visual_analysis)
            for phrase in skip_phrases:
                if visual_analysis.lower().startswith(phrase.lower()):
                    visual_analysis = visual_analysis[len(phrase):].strip()
                    if visual_analysis and visual_analysis[0] in ',:':
                        visual_analysis = visual_analysis[1:].strip()
                    if visual_analysis:
                        visual_analysis = visual_analysis[0].upper() + visual_analysis[1:]
                    break
            
            if len(visual_analysis) < original_length * 0.3:
                visual_analysis = result.get('response', '').strip()

            # Step 2: Get additional context/info if it's a product
            if self.is_likely_product(visual_analysis) or (detected_class and detected_class in ['cell phone', 'laptop', 'mouse', 'keyboard']):
                print("   💡 Fetching product details and pricing...")
                additional_info = self.get_product_context(visual_analysis)
                if additional_info:
                    return f"🔍 Visual Analysis:\n{visual_analysis}\n\n💰 Product Info:\n{additional_info}"
            
            return f"🔍 Analysis:\n{visual_analysis}"

        except Exception as e:
            print(f"❌ AI Error: {e}")
            import traceback
            traceback.print_exc()
            return f"Error during analysis: {str(e)}\n\nCheck that Ollama is running: ollama list"

    def is_likely_product(self, text):
        """Detect if the analysis is describing a product."""
        product_keywords = [
            'headphone', 'mouse', 'keyboard', 'phone', 'laptop', 'watch',
            'camera', 'speaker', 'earbuds', 'tablet', 'monitor', 'controller',
            'shoe', 'bag', 'bottle', 'glasses', 'device', 'gadget', 'brand',
            'model', 'sony', 'apple', 'samsung', 'logitech', 'nike', 'adidas'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in product_keywords)

    def get_product_context(self, description):
        """Get pricing and specifications for identified products."""
        try:
            # Extract key product info from description
            desc_lower = description.lower()
            
            prompt = f"""You are a product expert. Based on this visual description, provide specific product information:

Description: "{description}"

Provide:
1. If you can identify the EXACT product name/model, state it clearly
2. Typical price range (be realistic)
3. Key specifications or features (2-3 main ones)
4. Who is this product for?

Format as a natural paragraph (3-4 sentences). Be factual and specific.

Example output:
"This appears to be the Sony WH-1000XM5, typically priced around $399. It features industry-leading noise cancellation, 30-hour battery life, and supports LDAC audio codec. Popular among frequent travelers and audiophiles who prioritize sound quality."

Now provide info for the product described above:"""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.text_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150,
                        "top_p": 0.85,
                    }
                },
                timeout=45
            )

            if response.status_code == 200:
                result = response.json()
                info = result.get('response', '').strip()
                
                # Clean up if it starts with phrases like "Based on..."
                cleanup = ["based on the description", "based on this", "according to", "from the description"]
                info_lower = info.lower()
                for phrase in cleanup:
                    if info_lower.startswith(phrase):
                        # Find first sentence after this phrase
                        sentences = info.split('.')
                        if len(sentences) > 1:
                            info = '.'.join(sentences[1:]).strip()
                            if info:
                                info = info[0].upper() + info[1:] if len(info) > 1 else info.upper()
                        break
                
                if info and len(info) > 20 and not info.lower().startswith("i "):
                    return info
            return None
        except Exception as e:
            print(f"   ⚠️ Product lookup error: {e}")
            return None

    def detect_objects(self, frame):
        """Detect objects in frame using YOLO."""
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        detections = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf > self.confidence_threshold:
                    cls_id = int(box.cls[0])
                    name = self.model.names[cls_id]
                    
                    bbox = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'name': name,
                        'confidence': conf,
                        'box': bbox
                    })
        
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        return detections[:15]

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
        """Draw description box with AI analysis."""
        height, width = frame.shape[:2]
        box_height = min(350, height // 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, height - box_height), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
        
        cv2.rectangle(frame, (0, height - box_height), (width, height), (0, 255, 255), 3)
        cv2.putText(frame, "AI ANALYSIS", (20, height - box_height + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Split into sections
        sections = text.split('\n\n')
        y_offset = height - box_height + 75
        
        for section in sections:
            if not section.strip():
                continue
            
            # Check if section has a special marker
            if section.startswith('💰 Product Info:'):
                cv2.putText(frame, "💰 Product Info:", (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 100), 2)
                y_offset += 30
                section = section.replace('💰 Product Info:', '').strip()
            
            # Word wrap
            words = section.split()
            lines = []
            current_line = []
            max_width = width - 50
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                
                if text_size < max_width:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            for line in lines:
                if y_offset > height - 20:
                    break
                cv2.putText(frame, line, (30, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 28
            
            y_offset += 12

    def draw_ui(self, frame):
        """Draw UI elements."""
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        
        mode_text = ""
        if self.analysis_mode == "full_frame":
            mode_text = " [FULL FRAME MODE]"
        
        instructions = f"SPACE: Analyze{mode_text} | F: Full Frame | S: Select Object | UP/DOWN: Navigate | Q: Quit"
        cv2.putText(frame, instructions, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def run(self):
        """Main loop."""
        video = cv2.VideoCapture(0)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        video.set(cv2.CAP_PROP_FPS, 30)
        
        if not video.isOpened():
            print("❌ Error: Could not open camera!")
            return

        print("\n🎮 Controls:")
        print("   [SPACE]    ➜ Analyze selected object OR full frame")
        print("   [F]        ➜ Toggle FULL FRAME analysis mode")
        print("   [S]        ➜ Toggle SELECT OBJECT mode")
        print("   [UP/DOWN]  ➜ Navigate between detected objects")
        print("   [Q]        ➜ Quit")
        print("\n💡 What can it analyze?")
        print("   👤 People - appearance, clothing, expression")
        print("   📱 Products - brand, model, price, specs")
        print("   🐕 Animals - species, breed, characteristics")
        print("   🍕 Food - dishes, ingredients")
        print("   🏞️  Places - locations, scenes")
        print("   📦 Objects - anything you show it!")
        print("\n🎥 Camera ready!\n")

        while True:
            ret, frame = video.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Always detect objects for reference
            self.all_detections = self.detect_objects(frame)
            
            self.draw_ui(frame)

            if self.analysis_mode == "full_frame":
                # Full frame mode - analyze everything
                if self.current_desc:
                    self.draw_description_box(frame, self.current_desc)
                else:
                    cv2.putText(frame, "FULL FRAME MODE - Press SPACE to analyze entire view", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                # Object selection mode
                if self.all_detections:
                    self.selected_detection_idx = min(
                        self.selected_detection_idx, 
                        len(self.all_detections) - 1
                    )
                    self.draw_boxes(frame)
                    det = self.all_detections[self.selected_detection_idx]
                    
                    if self.current_desc:
                        self.draw_description_box(frame, self.current_desc)
                    else:
                        hint = f"Selected: {det['name']} - Press SPACE to analyze!"
                        cv2.putText(frame, hint, (20, 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, "No objects detected - Press F for full frame analysis", 
                               (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            if self.is_loading:
                height = frame.shape[0]
                cv2.putText(frame, "🧠 AI ANALYZING...", (20, height - 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            cv2.imshow("Universal Smart Analyzer", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
                
            elif key == ord('f') or key == ord('F'):
                # Toggle full frame mode
                self.analysis_mode = "full_frame" if self.analysis_mode != "full_frame" else "auto"
                self.current_desc = None
                mode_name = "FULL FRAME" if self.analysis_mode == "full_frame" else "OBJECT SELECT"
                print(f"\n🔄 Switched to {mode_name} mode")
                
            elif key == ord('s') or key == ord('S'):
                # Toggle selection mode
                self.analysis_mode = "auto"
                self.current_desc = None
                print(f"\n🔄 Switched to OBJECT SELECT mode")
                    
            elif key == ord(' '):
                if not self.is_loading:
                    self.is_loading = True
                    
                    if self.analysis_mode == "full_frame":
                        print(f"\n🔍 Analyzing full frame...")
                        desc = self.analyze_with_ai(frame, box=None, detected_class=None)
                    elif self.all_detections:
                        det = self.all_detections[self.selected_detection_idx]
                        print(f"\n🔍 Analyzing {det['name']}...")
                        desc = self.analyze_with_ai(frame, det['box'], det['name'])
                    else:
                        print(f"\n🔍 No objects detected, analyzing full frame...")
                        desc = self.analyze_with_ai(frame, box=None, detected_class=None)
                    
                    self.current_desc = desc
                    self.is_loading = False
                    print(f"✅ Analysis complete!\n")
                    
            elif key == 82 or key == 0:  # UP
                if self.all_detections and self.analysis_mode != "full_frame":
                    self.selected_detection_idx = (self.selected_detection_idx - 1) % len(self.all_detections)
                    self.current_desc = None
                    
            elif key == 84 or key == 1:  # DOWN
                if self.all_detections and self.analysis_mode != "full_frame":
                    self.selected_detection_idx = (self.selected_detection_idx + 1) % len(self.all_detections)
                    self.current_desc = None

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔮 UNIVERSAL SMART ANALYZER 🔮")
    print("=" * 60)
    print("\n🌟 Powered by AI Vision + Language Models")
    print("\n💫 Can Analyze:")
    print("   👤 People - Appearance, clothing, age, expression")
    print("   📱 Products - Brand, model, price, specifications")
    print("   🐕 Animals - Species, breed, characteristics")
    print("   🍕 Food - Dishes, ingredients, cuisine")
    print("   🏞️  Scenes - Locations, environments")
    print("   🎨 Art - Style, medium, composition")
    print("   🚗 Vehicles - Make, model, year")
    print("   📦 ANY Object - Detailed identification!")
    print("\n📋 Setup Required:")
    print("   1. ollama pull llava")
    print("   2. ollama pull llama3.2")
    print("\n🎯 Two Analysis Modes:")
    print("   • Object Mode: Detect and analyze specific items")
    print("   • Full Frame: Analyze entire camera view")
    print("=" * 60 + "\n")
    
    try:
        analyzer = UniversalSmartAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")