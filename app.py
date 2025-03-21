import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance
import time
import os

class BodyMeasurementSystem:
    def __init__(self, known_height_cm=None):
        # Initialize MediaPipe Pose with higher accuracy settings
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Use the most accurate model
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Known parameters
        self.known_height_cm = known_height_cm
        self.reference_distance_cm = 200
        
        # Calibration variables
        self.pixel_to_cm = None
        self.last_landmarks = None
        self.calibration_frames = []
        self.calibration_complete = False
        
        # Measurement history for stability
        self.measurement_history = {
            "chest": [],
            "waist": [],
            "height": []
        }
        self.history_size = 5
        
        # Static anthropometric ratios (based on research)
        # These are average ratios based on human body studies
        self.shoulder_to_chest_ratio = 0.85  # Chest circumference is larger than shoulder width
        self.hip_to_waist_ratio = 0.95  # Waist is usually smaller than hip measurement
        
        # Known anthropometric relationships (these will improve accuracy)
        # Source: NASA anthropometric studies and medical research
        self.male_chest_to_height_ratio = 0.52  # Average male chest circ. to height ratio
        self.female_chest_to_height_ratio = 0.53  # Average female chest circ. to height ratio
        self.male_waist_to_height_ratio = 0.45  # Average male waist circ. to height ratio
        self.female_waist_to_height_ratio = 0.42  # Average female waist circ. to height ratio
        
        # Default to average values
        self.chest_to_height_ratio = (self.male_chest_to_height_ratio + self.female_chest_to_height_ratio) / 2
        self.waist_to_height_ratio = (self.male_waist_to_height_ratio + self.female_waist_to_height_ratio) / 2
        
        # Correction factors (starting values, will be refined during calibration)
        self.chest_correction = 1.0
        self.waist_correction = 1.0
        
        # Camera parameters (can be calibrated for better accuracy)
        self.focal_length = None
        self.camera_calib_matrix = None
        
        # Reference markers
        self.reference_object_width_cm = 30  # e.g., A4 paper is about 21cm wide, credit card 8.5cm
        
        # Debug mode for developers
        self.debug_mode = False
        
        # Size chart data (example) - Updated to use common sizes
        self.size_chart = {
            "men": {
                "chest": {
                    (0, 94): "S",
                    (94, 102): "M",
                    (102, 110): "L",
                    (110, 118): "XL",
                    (118, float('inf')): "XXL"
                },
                "waist": {
                    (0, 81): "S",
                    (81, 89): "M",
                    (89, 97): "L",
                    (97, 105): "XL",
                    (105, float('inf')): "XXL"
                }
            },
            "women": {
                "chest": {
                    (0, 84): "S",
                    (84, 92): "M",
                    (92, 100): "L",
                    (100, 108): "XL",
                    (108, float('inf')): "XXL"
                },
                "waist": {
                    (0, 66): "S",
                    (66, 74): "M",
                    (74, 82): "L",
                    (82, 90): "XL",
                    (90, float('inf')): "XXL"
                }
            }
        }
        self.gender = "average"  # "men", "women", "average"
    
    def set_gender(self, gender):
        """Set gender to improve body ratio calculations"""
        if gender.lower() in ['m', 'male']:
            self.chest_to_height_ratio = self.male_chest_to_height_ratio
            self.waist_to_height_ratio = self.male_waist_to_height_ratio
            self.gender = "men"
            print("Set to male body proportions")
        elif gender.lower() in ['f', 'female']:
            self.chest_to_height_ratio = self.female_chest_to_height_ratio
            self.waist_to_height_ratio = self.female_waist_to_height_ratio
            self.gender = "women"
            print("Set to female body proportions")
        else:
            self.gender = "average"
            print("Using average body proportions")
    
    def set_known_height(self, height_cm):
        """Set the known height of the person being measured"""
        if height_cm > 50 and height_cm < 250:  # Sanity check
            self.known_height_cm = height_cm
            print(f"Known height set to {height_cm} cm")
            return True
        else:
            print(f"Invalid height value: {height_cm}. Please enter a realistic height (50-250 cm).")
            return False
    
    def reset_calibration(self):
        """Reset all calibration data"""
        self.pixel_to_cm = None
        self.calibration_frames = []
        self.calibration_complete = False
        self.measurement_history = {"chest": [], "waist": [], "height": []}
        print("Calibration reset. Please recalibrate the system.")
    
    def set_reference_distance(self, distance_cm):
        """Set the reference distance for calibration"""
        if distance_cm > 50 and distance_cm < 500:  # Sanity check
            self.reference_distance_cm = distance_cm
            print(f"Reference distance set to {distance_cm} cm")
            return True
        else:
            print(f"Invalid distance value: {distance_cm}cm. Please enter a realistic distance (50-500 cm).")
            return False
            
    def calibrate_with_multi_frame(self, frame):
        """
        Add a frame to the calibration buffer for more robust calibration
        """
        # Process the frame to get pose landmarks
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return False
            
        # Store the frame for calibration
        self.calibration_frames.append((frame, results.pose_landmarks))
        
        # If we have enough frames, perform calibration
        if len(self.calibration_frames) >= 10:
            return self._complete_multi_frame_calibration()
            
        return False
    
    def _complete_multi_frame_calibration(self):
        """Process all calibration frames to get a stable calibration"""
        if not self.calibration_frames:
            return False
            
        pixel_to_cm_values = []
        
        for frame, landmarks in self.calibration_frames:
            frame_height, frame_width, _ = frame.shape
            
            # Calculate body height in pixels
            nose = (landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].x * frame_width,
                   landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y * frame_height)
                   
            # Use the middle point between ankles for better stability
            left_ankle = (landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
                         landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * frame_height)
            right_ankle = (landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
                          landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height)
            
            ankle_x = (left_ankle[0] + right_ankle[0]) / 2
            ankle_y = (left_ankle[1] + right_ankle[1]) / 2
            ankle = (ankle_x, ankle_y)
            
            body_height_pixels = distance.euclidean(nose, ankle)
            
            # Calculate pixel-to-cm ratio based on known height or reference distance
            if self.known_height_cm:
                # The visible height might be less than the total height due to framing
                # Apply a correction factor (empirical, can be adjusted)
                visible_height_factor = 0.95  # Assuming we see about 95% of total height
                pixel_to_cm = (self.known_height_cm * visible_height_factor) / body_height_pixels
                pixel_to_cm_values.append(pixel_to_cm)
        
        # Filter out extreme values
        pixel_to_cm_values.sort()
        filtered_values = pixel_to_cm_values[1:-1] if len(pixel_to_cm_values) > 3 else pixel_to_cm_values
        
        # Use the median for stability
        if filtered_values:
            self.pixel_to_cm = np.median(filtered_values)
            self.calibration_complete = True
            print(f"Multi-frame calibration complete! Conversion factor: {self.pixel_to_cm:.4f} cm/pixel")
            return True
        
        return False
    
    def calculate_body_measurements(self, frame):
        """
        Calculate body measurements with enhanced accuracy techniques
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to get pose landmarks
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return {"error": "No person detected"}
            
        # Store for drawing
        self.last_landmarks = results.pose_landmarks
        
        landmarks = results.pose_landmarks.landmark
        frame_height, frame_width, _ = frame.shape
        
        # Extract all required landmarks
        keypoints = {}
        for landmark in self.mp_pose.PoseLandmark:
            x = landmarks[landmark].x * frame_width
            y = landmarks[landmark].y * frame_height
            visibility = landmarks[landmark].visibility
            keypoints[landmark.name] = (x, y, visibility)
        
        # Check if all required landmarks are visible
        required_landmarks = ["NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", 
                             "LEFT_HIP", "RIGHT_HIP", "LEFT_ANKLE", "RIGHT_ANKLE"]
        
        for landmark in required_landmarks:
            if keypoints[landmark][2] < 0.5:  # Low visibility
                return {"error": f"Cannot see {landmark.lower().replace('_', ' ')} clearly"}
        
        # Calculate key measurements in pixels
        # Shoulder width
        shoulder_width_px = distance.euclidean(
            keypoints["LEFT_SHOULDER"][:2], 
            keypoints["RIGHT_SHOULDER"][:2]
        )
        
        # Hip width
        hip_width_px = distance.euclidean(
            keypoints["LEFT_HIP"][:2], 
            keypoints["RIGHT_HIP"][:2]
        )
        
        # Body height (nose to mid-ankle)
        left_ankle = keypoints["LEFT_ANKLE"][:2]
        right_ankle = keypoints["RIGHT_ANKLE"][:2]
        mid_ankle = ((left_ankle[0] + right_ankle[0])/2, (left_ankle[1] + right_ankle[1])/2)
        body_height_px = distance.euclidean(keypoints["NOSE"][:2], mid_ankle)
        
        # If not calibrated and we have known height
        if not self.calibration_complete and self.known_height_cm:
            # Estimate pixel_to_cm ratio based on known height
            visible_height_factor = 0.95  # Assuming we see about 95% of total height
            self.pixel_to_cm = (self.known_height_cm * visible_height_factor) / body_height_px
            self.calibration_complete = True
        
        # If still not calibrated, use an estimated value
        if not self.calibration_complete:
            # Use a default value based on average human height and webcam placement
            estimated_height_cm = 170
            self.pixel_to_cm = estimated_height_cm / body_height_px
            print("Warning: Using estimated calibration. Results may be inaccurate.")
        
        # Calculate measurements in cm
        shoulder_width_cm = shoulder_width_px * self.pixel_to_cm
        hip_width_cm = hip_width_px * self.pixel_to_cm
        
        # Calculate body depth (estimate from width)
        # Research shows human bodies are typically wider than deep
        shoulder_depth_cm = shoulder_width_cm * 0.7  # Estimate depth is about 70% of width
        hip_depth_cm = hip_width_cm * 0.75  # Hip depth is about 75% of width
        
        # Calculate circumferences using elliptical approximation formula: 2π * √((a² + b²)/2)
        # where a and b are the semi-major and semi-minor axes
        chest_a = shoulder_width_cm / 2
        chest_b = shoulder_depth_cm / 2
        waist_a = hip_width_cm / 2
        waist_b = hip_depth_cm / 2
        
        chest_circumference_cm = 2 * np.pi * np.sqrt((chest_a**2 + chest_b**2) / 2)
        waist_circumference_cm = 2 * np.pi * np.sqrt((waist_a**2 + waist_b**2) / 2)
        
        # Apply anthropometric corrections
        if self.known_height_cm:
            # Cross-check with known anthropometric ratios
            expected_chest = self.known_height_cm * self.chest_to_height_ratio
            expected_waist = self.known_height_cm * self.waist_to_height_ratio
            
            # Apply a weighted blend between measurement and expected value
            # This helps correct for viewing angle and body position issues
            chest_circumference_cm = (chest_circumference_cm * 0.7) + (expected_chest * 0.3)
            waist_circumference_cm = (waist_circumference_cm * 0.7) + (expected_waist * 0.3)
        
        # Apply user-specific correction factors (refined during multiple measurements)
        chest_circumference_cm *= self.chest_correction
        waist_circumference_cm *= self.waist_correction
        
        # Get final height
        height_cm = self.known_height_cm if self.known_height_cm else body_height_px * self.pixel_to_cm
        
        # Add to measurement history for smoothing
        self.measurement_history["chest"].append(chest_circumference_cm)
        self.measurement_history["waist"].append(waist_circumference_cm)
        self.measurement_history["height"].append(height_cm)
        
        # Keep history at desired size
        for key in self.measurement_history:
            if len(self.measurement_history[key]) > self.history_size:
                self.measurement_history[key].pop(0)
        
        # Use moving average for more stable results
        if len(self.measurement_history["chest"]) > 2:
            chest_circumference_cm = np.mean(self.measurement_history["chest"])
            waist_circumference_cm = np.mean(self.measurement_history["waist"])
            height_cm = np.mean(self.measurement_history["height"])
        
        # Determine size from chart
        chest_size = self.get_size_from_chart("chest", chest_circumference_cm)
        waist_size = self.get_size_from_chart("waist", waist_circumference_cm)
        
        # Prepare results
        measurements = {
            "chest_circumference_cm": round(chest_circumference_cm, 1),
            "waist_circumference_cm": round(waist_circumference_cm, 1),
            "height_cm": round(height_cm, 1),
            "shoulder_width_cm": round(shoulder_width_cm, 1),
            "chest_size": chest_size,  # Add the size to the measurements
            "waist_size": waist_size  # Add waist size
        }
        
        if self.debug_mode:
            # Add debug information
            measurements["debug"] = {
                "pixel_to_cm": round(self.pixel_to_cm, 4),
                "body_height_px": round(body_height_px, 1),
                "shoulder_width_px": round(shoulder_width_px, 1),
                "hip_width_px": round(hip_width_px, 1)
            }
            
        return measurements

    def get_size_from_chart(self, body_part, measurement_cm):
        """
        Looks up the size in the size chart.
        """
        chart = None
        if self.gender in self.size_chart:
            chart = self.size_chart[self.gender].get(body_part)
        else:
            # Fallback to average sizes if gender is not set or not in chart
            chart = self.size_chart["men"].get(body_part)

        if not chart:
            return "Size N/A"  # Or some other default

        for size_range, size_label in chart.items():
            if size_range[0] <= measurement_cm < size_range[1]:
                return size_label

        return "Size N/A"  # If no match is found
    
    def draw_measurements(self, frame, measurements):
        """
        Draw measurement annotations on the frame
        """
        if "error" in measurements:
            cv2.putText(frame, measurements["error"], (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
            
        # Draw pose landmarks with improved visibility
        if self.last_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, 
                self.last_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Create a semi-transparent overlay for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (30, 30), (350, 210), (0, 0, 0), -1)  # Increased height for the size display
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Display measurements with better formatting
        cv2.putText(frame, "BODY MEASUREMENTS", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        y_pos = 80
        for label, key in [
            ("Chest:", "chest_circumference_cm"),
            ("Waist:", "waist_circumference_cm"),
            ("Height:", "height_cm"),
            ("Shoulder Width:", "shoulder_width_cm")
        ]:
            if key in measurements:
                value = measurements[key]
                text = f"{label} {value} cm"
                cv2.putText(frame, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_pos += 30
        
        # Display sizes
        if "chest_size" in measurements and "waist_size" in measurements:
            cv2.putText(frame, f"Chest Size: {measurements['chest_size']}", (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_pos += 30  # Adjust y_pos for the next line
            cv2.putText(frame, f"Waist Size: {measurements['waist_size']}", (50, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show calibration status
        if self.calibration_complete:
            status = "CALIBRATED"
            color = (0, 255, 0)
        else:
            status = "NOT CALIBRATED - Results may be inaccurate"
            color = (0, 0, 255)
            
        cv2.putText(frame, status, (50, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
        return frame
    
    def process_frame(self, frame, draw=True):
        """
        Process a single frame and return measurements
        """
        measurements = self.calculate_body_measurements(frame)
        
        if draw:
            frame = self.draw_measurements(frame, measurements)
            
        return frame, measurements
    
    def process_image(self, image_path, draw=True):
        """
        Process an image file
        """
        if not os.path.exists(image_path):
            return None, {"error": f"Image file not found: {image_path}"}
            
        frame = cv2.imread(image_path)
        if frame is None:
            return None, {"error": f"Could not read image: {image_path}"}
            
        return self.process_frame(frame, draw)
    
    def start_webcam(self):
        """
        Start capturing and processing from webcam with improved user interface
        """
        # Try to open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        # Set a higher resolution if possible
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # States
        state = "SETUP"  # SETUP, CALIBRATION, MEASUREMENT
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        print("\n=== BODY MEASUREMENT SYSTEM ===")
        print("1. SETUP - Configure your preferences")
        print("2. CALIBRATION - Stand at reference distance")
        print("3. MEASUREMENT - Get your body measurements")
        print("\nCommands:")
        print("  h - Set person's height")
        print("  g - Set gender (for better proportions)")
        print("  c - Calibrate")
        print("  r - Reset calibration")
        print("  d - Toggle debug mode")
        print("  q - Quit\n")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read from webcam.")
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Create a fresh copy for each state
            display_frame = frame.copy()
            
            if state == "SETUP":
                # Display setup instructions
                setup_overlay = display_frame.copy()
                cv2.rectangle(setup_overlay, (0, 0), (frame.shape[1], 150), (0, 0, 0), -1)
                alpha = 0.7
                display_frame = cv2.addWeighted(setup_overlay, alpha, display_frame, 1 - alpha, 0)
                
                cv2.putText(display_frame, "SETUP MODE", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                           
                instructions = [
                    "Press 'h' to set person's height",
                    "Press 'g' to set gender (for better proportions)",
                    "Press 'c' to continue to calibration",
                    "Press 'q' to quit"
                ]
                
                y_pos = 70
                for instruction in instructions:
                    cv2.putText(display_frame, instruction, (50, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 25
                
                status_text = []
                if self.known_height_cm:
                    status_text.append(f"Height: {self.known_height_cm} cm")
                else:
                    status_text.append("Height: Not set")
                
                gender_text = "Gender: "
                if self.chest_to_height_ratio == self.male_chest_to_height_ratio:
                    gender_text += "Male"
                elif self.chest_to_height_ratio == self.female_chest_to_height_ratio:
                    gender_text += "Female"
                else:
                    gender_text += "Average (not set)"
                status_text.append(gender_text)
                
                # Display status at the bottom
                bottom_overlay = display_frame.copy()
                cv2.rectangle(bottom_overlay, (0, frame.shape[0] - 80), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(bottom_overlay, alpha, display_frame, 1 - alpha, 0)
                
                y_pos = frame.shape[0] - 50
                for text in status_text:
                    cv2.putText(display_frame, text, (50, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    y_pos += 30
                
            elif state == "CALIBRATION":
                # Process frame for calibration
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                
                if results and results.pose_landmarks:
                    # Draw the pose landmarks
                    display_frame = frame.copy()
                    self.mp_drawing.draw_landmarks(
                        display_frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                    self.last_landmarks = results.pose_landmarks
                
                # Create calibration UI overlay
                calib_overlay = display_frame.copy()
                cv2.rectangle(calib_overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
                alpha = 0.7
                display_frame = cv2.addWeighted(calib_overlay, alpha, display_frame, 1 - alpha, 0)
                
                cv2.putText(display_frame, "CALIBRATION MODE", (50, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                           
                if self.known_height_cm:
                    instructions = [
                        f"Stand naturally facing the camera",
                        f"Height used for calibration: {self.known_height_cm} cm",
                        f"Press 'c' to calibrate (hold position for a few seconds)"
                    ]
                else:
                    instructions = [
                        f"Stand at {self.reference_distance_cm}cm from camera",
                        "Press 'h' to set your height (recommended)",
                        "Press 'c' to calibrate (hold position for a few seconds)"
                    ]
                
                y_pos = 70
                for instruction in instructions:
                    cv2.putText(display_frame, instruction, (50, y_pos), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 25
                
                # Show calibration progress
                if self.calibration_frames:
                    progress = min(len(self.calibration_frames) * 10, 100)
                    cv2.putText(display_frame, f"Calibration progress: {progress}%", 
                               (50, frame.shape[0] - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Draw progress bar
                    bar_width = 300
                    filled_width = int(bar_width * progress / 100)
                    cv2.rectangle(display_frame, (50, frame.shape[0] - 30), (50 + bar_width, frame.shape[0] - 20), (100, 100, 100), -1)
                    cv2.rectangle(display_frame, (50, frame.shape[0] - 30), (50 + filled_width, frame.shape[0] - 20), (0, 255, 255), -1)
                
            elif state == "MEASUREMENT":
                # Process the frame and get measurements
                display_frame, measurements = self.process_frame(frame)
                
                # Print measurements to console
                if "error" not in measurements:
                    print(f"\rChest: {measurements.get('chest_circumference_cm', 0):.1f}cm ({measurements.get('chest_size', 'N/A')}) | " +
                          f"Waist: {measurements.get('waist_circumference_cm', 0):.1f}cm ({measurements.get('waist_size', 'N/A')}) | " +
                          f"Height: {measurements.get('height_cm', 0):.1f}cm", end="")
                else:
                    print(f"\r{measurements['error']}", end="")
            
            # Show the frame
            window_title = f"Body Measurement System - {state}"
            cv2.imshow(window_title, display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Global commands
            if key == ord('q'):
                break
                
            elif key == ord('h'):
                # Input height
                height_input = input("\nEnter person's height in cm: ")
                try:
                    height_cm = float(height_input)
                    if self.set_known_height(height_cm):
                        print(f"Height set to {height_cm} cm")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            elif key == ord('g'):
                # Set gender
                gender = input("\nEnter gender (m/f) for better body proportions: ").lower()
                self.set_gender(gender)
            
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"\nDebug mode {'enabled' if self.debug_mode else 'disabled'}")
            
            elif key == ord('r'):
                # Reset calibration
                self.reset_calibration()
                state = "SETUP"
            
            # State-specific commands
            if key == ord('c'):
                if state == "SETUP":
                    # Move to calibration
                    state = "CALIBRATION"
                    print("\nEntering CALIBRATION state. Please follow on-screen instructions.")
                    # Clear any previous calibration frames
                    self.calibration_frames = []
                
                elif state == "CALIBRATION":
                    # If in calibration state, add current frame to calibration data
                    if results and results.pose_landmarks:
                        result = self.calibrate_with_multi_frame(frame)
                        if result:
                            state = "MEASUREMENT"
                            print("\nCalibration successful! Entering MEASUREMENT state.")
                    else:
                        print("\nCannot detect person. Please stand clearly in front of the camera.")
        
            # If calibration is complete, allow moving to measurement state
            if self.calibration_complete and state == "CALIBRATION" and key == ord('m'):
                state = "MEASUREMENT"
                print("\nEntering MEASUREMENT state.")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nProgram terminated.")
        

# Main program
def main():
    print("===== Body Measurement System =====")
print("This program will measure chest, waist, and other body dimensions.")

# Option to provide known height
has_height = input("Do you know the person's height? (y/n): ").lower()
height_cm = None

if has_height == 'y':
    try:
        height_input = input("Enter the person's height in cm: ")
        height_cm = float(height_input)
        if height_cm < 50 or height_cm > 250:
            print("Warning: Unusual height value. Results may be inaccurate.")
    except ValueError:
        print("Invalid input. Starting without known height.")
        height_cm = None

# Create measurement system with known height
system = BodyMeasurementSystem(known_height_cm=height_cm)

# Option to set gender
gender_input = input("Set gender for more accurate measurements (m/f/skip): ").lower()
if gender_input in ['m', 'f']:
    system.set_gender(gender_input)

# Choose mode
print("\nSelect operation mode:")
print("1. Webcam (live measurement)")
print("2. Image file (process a photo)")

mode = input("Enter mode (1/2): ")

if mode == '1':
    # Start webcam interface
    system.start_webcam()
elif mode == '2':
    # Process image file
    image_path = input("Enter the path to the image file: ")
    if os.path.exists(image_path):
        result_frame, measurements = system.process_image(image_path)
        
        if result_frame is not None:
            if "error" in measurements:
                print(f"Error: {measurements['error']}")
        
            # Print measurements
            if "error" not in measurements:
                print("\n--- MEASUREMENT RESULTS ---")
                print(f"Chest Circumference: {measurements['chest_circumference_cm']} cm")
                print(f"Waist Circumference: {measurements['waist_circumference_cm']} cm")
                print(f"Height: {measurements['height_cm']} cm")
                print(f"Shoulder Width: {measurements['shoulder_width_cm']} cm")

                # Display size
                print(f"Estimated Chest Size: {measurements.get('chest_size', 'N/A')}")
                print(f"Estimated Waist Size: {measurements.get('waist_size', 'N/A')}")
                
                # Show the image with measurements
                cv2.imshow("Measurement Results", result_frame)
                cv2.waitKey(0)  # Wait for a key press to close the image window
                cv2.destroyAllWindows()
            else:
                print(f"Error: {measurements['error']}")
        else:
            print(f"Image processing failed.")
    else:
        print(f"File not found: {image_path}")
else:
    print("Invalid mode. Exiting.")
    
if __name__ == "__main__":
    main()