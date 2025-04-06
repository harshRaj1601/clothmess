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
        
        # UI State
        self.state = "SETUP"  # SETUP, CALIBRATION, MEASUREMENT
        self.message = ""
        self.message_color = (255, 255, 255)
        self.message_timer = 0
        self.input_active = False
        self.input_text = ""
        self.input_type = None  # "height", "gender"
        
    def set_gender(self, gender):
        """Set gender to improve body ratio calculations"""
        if gender.lower() in ['m', 'male']:
            self.chest_to_height_ratio = self.male_chest_to_height_ratio
            self.waist_to_height_ratio = self.male_waist_to_height_ratio
            self.gender = "men"
            self.show_message("Set to male body proportions", color=(0, 255, 0))
            return True
        elif gender.lower() in ['f', 'female']:
            self.chest_to_height_ratio = self.female_chest_to_height_ratio
            self.waist_to_height_ratio = self.female_waist_to_height_ratio
            self.gender = "women"
            self.show_message("Set to female body proportions", color=(0, 255, 0))
            return True
        else:
            self.gender = "average"
            self.show_message("Using average body proportions", color=(0, 255, 255))
            return False
    
    def set_known_height(self, height_cm):
        """Set the known height of the person being measured"""
        try:
            height_cm = float(height_cm)
            if height_cm > 50 and height_cm < 250:  # Sanity check
                self.known_height_cm = height_cm
                self.show_message(f"Height set to {height_cm} cm", color=(0, 255, 0))
                return True
            else:
                self.show_message(f"Invalid height: {height_cm}. Enter 50-250 cm.", color=(0, 0, 255))
                return False
        except ValueError:
            self.show_message("Invalid input. Please enter a number.", color=(0, 0, 255))
            return False
    
    def reset_calibration(self):
        """Reset all calibration data"""
        self.pixel_to_cm = None
        self.calibration_frames = []
        self.calibration_complete = False
        self.measurement_history = {"chest": [], "waist": [], "height": []}
        self.show_message("Calibration reset. Please recalibrate.", color=(0, 255, 255))
    
    def set_reference_distance(self, distance_cm):
        """Set the reference distance for calibration"""
        if distance_cm > 50 and distance_cm < 500:  # Sanity check
            self.reference_distance_cm = distance_cm
            self.show_message(f"Reference distance set to {distance_cm} cm", color=(0, 255, 0))
            return True
        else:
            self.show_message(f"Invalid distance: {distance_cm}. Enter 50-500 cm.", color=(0, 0, 255))
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
            self.show_message(f"Calibration complete! Conversion factor: {self.pixel_to_cm:.4f} cm/pixel", color=(0, 255, 0))
            return True
        
        return False
    
    def calculate_body_measurements(self, frame):
        """
        Calculate body measurements with enhanced accuracy techniques
        """
        # Check if pixel_to_cm is None
        if self.pixel_to_cm is None:
            if self.known_height_cm:
                # If known height is set but calibration is incomplete
                frame_height, frame_width, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)

                if results and results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    nose = (landmarks[self.mp_pose.PoseLandmark.NOSE].x * frame_width,
                           landmarks[self.mp_pose.PoseLandmark.NOSE].y * frame_height)

                    left_ankle = (landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * frame_width,
                                 landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * frame_height)
                    right_ankle = (landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * frame_width,
                                  landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * frame_height)

                    ankle_x = (left_ankle[0] + right_ankle[0]) / 2
                    ankle_y = (left_ankle[1] + right_ankle[1]) / 2
                    ankle = (ankle_x, ankle_y)

                    body_height_pixels = distance.euclidean(nose, ankle)
                    visible_height_factor = 0.95
                    self.pixel_to_cm = (self.known_height_cm * visible_height_factor) / body_height_pixels
                    self.calibration_complete = True #Consider it as calibrated as we are estimating pixel to cm from height

                else:
                     return {"error": "No person detected for auto-calibration"} # Can't calibrate, no people seen even to estimate

            else:
                self.show_message("Please set height for estimation or calibrate", color=(0, 0, 255))
                return {"error": "pixel_to_cm is None and no known height set"}

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
    
    def show_message(self, text, color=(255, 255, 255), duration=3):
        """Show a message with timer"""
        self.message = text
        self.message_color = color
        self.message_timer = time.time() + duration
    
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
        # Using a darker panel style with gradient for modern look
        overlay = frame.copy()
        
        # Panel gradient background
        h, w = frame.shape[:2]
        panel_w = 350
        panel_h = 220
        panel_x = 30
        panel_y = 30
        
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + 50), (60, 60, 60), -1)  # Header area
        
        # Blend overlay with original frame
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Add accent line
        cv2.line(frame, (panel_x, panel_y + 50), (panel_x + panel_w, panel_y + 50), (0, 165, 255), 2)
        
        # Display measurements with better formatting
        cv2.putText(frame, "BODY MEASUREMENTS", (panel_x + 20, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        y_pos = panel_y + 80
        for label, key, color in [
            ("Chest:", "chest_circumference_cm", (102, 204, 255)),
            ("Waist:", "waist_circumference_cm", (102, 255, 178)),
            ("Height:", "height_cm", (178, 102, 255)),
            ("Shoulder Width:", "shoulder_width_cm", (255, 178, 102))
        ]:
            if key in measurements:
                value = measurements[key]
                text = f"{label} {value} cm"
                cv2.putText(frame, text, (panel_x + 20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
                y_pos += 30
        
        # Display sizes in a highlighted area
        size_panel_y = y_pos
        cv2.rectangle(overlay, (panel_x + 10, size_panel_y - 20), (panel_x + panel_w - 10, size_panel_y + 60), (60, 60, 60), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        if "chest_size" in measurements and "waist_size" in measurements:
            # Create stylish size indicators
            chest_size = measurements['chest_size']
            waist_size = measurements['waist_size']
            
            # Size display in modern UI style
            cv2.putText(frame, "Recommended Sizes:", (panel_x + 20, size_panel_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
            
            # Chest size in a pill shape
            size_x = panel_x + 20
            size_y = size_panel_y + 25
            
            # Pills for chest size
            pill_width = 50
            pill_height = 24
            cv2.rectangle(frame, (size_x, size_y - pill_height//2), 
                         (size_x + pill_width, size_y + pill_height//2), 
                         (102, 204, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, f"Chest: {chest_size}", (size_x + 5, size_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Pills for waist size
            size_x = panel_x + 180
            cv2.rectangle(frame, (size_x, size_y - pill_height//2), 
                         (size_x + pill_width, size_y + pill_height//2), 
                         (102, 255, 178), -1, cv2.LINE_AA)
            cv2.putText(frame, f"Waist: {waist_size}", (size_x + 5, size_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Show calibration status with icon
        if self.calibration_complete:
            status = "CALIBRATED"
            color = (0, 255, 0)
            icon = "✓"
        else:
            status = "NOT CALIBRATED"
            color = (0, 0, 255)
            icon = "!"
            
        # Status in modern UI style
        status_x = w - 200
        status_y = 40
        cv2.rectangle(frame, (status_x - 10, status_y - 30), (status_x + 180, status_y + 10), (40, 40, 40), -1)
        cv2.putText(frame, f"{icon} {status}", (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show debug info if enabled
        if self.debug_mode and "debug" in measurements:
            debug_y = h - 100
            debug_info = measurements["debug"]
            cv2.rectangle(frame, (w - 300, debug_y - 20), (w - 20, debug_y + 80), (40, 40, 40), -1)
            cv2.putText(frame, "DEBUG INFO", (w - 290, debug_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            y_offset = 20
            for key, value in debug_info.items():
                cv2.putText(frame, f"{key}: {value}", (w - 290, debug_y + y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_offset += 20
                
        return frame
    
    def draw_setup_screen(self, frame):
        """Draw the setup screen UI"""
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay for better contrast
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)  # Top banner
        
        # Create a modern panel for instructions
        panel_w = w - 100
        panel_h = 250
        panel_x = 50
        panel_y = 180
        
        # Main panel
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + 50), (60, 60, 60), -1)  # Header
        
        # Status panel
        status_panel_y = panel_y + panel_h + 30
        cv2.rectangle(overlay, (panel_x, status_panel_y), (panel_x + panel_w, status_panel_y + 100), (40, 40, 40), -1)
        
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Modern UI styling
        cv2.line(frame, (panel_x, panel_y + 50), (panel_x + panel_w, panel_y + 50), (0, 165, 255), 2)  # Accent line
        
        # Draw title and subtitle
        cv2.putText(frame, "BODY MEASUREMENT SYSTEM", (50, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, "Setup Mode - Configure Your Preferences", (50, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Panel header
        cv2.putText(frame, "SETUP INSTRUCTIONS", (panel_x + 20, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions with icons
        instructions = [
            ("H", "Set person's height for accurate measurements"),
            ("G", "Set gender for better body proportions"),
            ("C", "Continue to calibration step"),
            ("D", "Toggle debug mode for detailed information"),
            ("Q", "Quit application")
        ]
        
        y_pos = panel_y + 90
        for key, text in instructions:
            # Key highlighting
            cv2.rectangle(frame, (panel_x + 20, y_pos - 20), (panel_x + 45, y_pos + 5), (60, 60, 60), -1)
            cv2.putText(frame, key, (panel_x + 25, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 1)
            
            # Instruction text
            cv2.putText(frame, text, (panel_x + 60, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
            y_pos += 40
        
        # Status panel header
        cv2.putText(frame, "CURRENT SETTINGS", (panel_x + 20, status_panel_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current settings
        settings = []
        if self.known_height_cm:
            settings.append(f"Height: {self.known_height_cm} cm")
        else:
            settings.append("Height: Not set")
        
        if self.gender == "men":
            settings.append("Gender: Male")
        elif self.gender == "women":
            settings.append("Gender: Female")
        else:
            settings.append("Gender: Not set (using average)")
        
        # Draw settings
        y_pos = status_panel_y + 70
        for setting in settings:
            cv2.putText(frame, setting, (panel_x + 30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
            y_pos += 30
        
        # Draw input prompt if active
        if self.input_active:
            self.draw_input_prompt(frame)
        
        return frame
    
    def draw_calibration_screen(self, frame):
        """Draw the calibration UI"""
        h, w = frame.shape[:2]
        
        # Overlay for instructions
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)  # Top banner
        
        # Calibration panel style
        panel_w = w - 100
        panel_h = 120
        panel_x = 50
        panel_y = 180
        
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + 50), (60, 60, 60), -1)  # Header
        
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        cv2.line(frame, (panel_x, panel_y + 50), (panel_x + panel_w, panel_y + 50), (0, 165, 255), 2)  # Accent line
        
        # Calibration Instructions
        cv2.putText(frame, "CALIBRATION MODE", (50, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
        cv2.putText(frame, "Stand facing the camera", (50, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # Panel Header
        cv2.putText(frame, "CALIBRATION INSTRUCTIONS", (panel_x + 20, panel_y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Instructions based on calibration
        if self.known_height_cm:
            instruction = "Stand naturally, Calibration will automatically start"
        else:
            instruction = "Stand at known distance, Calibration will start automatically"
        
        cv2.putText(frame, instruction, (panel_x + 20, panel_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1)
        
        # Show calibration progress
        progress = min(len(self.calibration_frames) * 10, 100)
        cv2.putText(frame, f"Calibration progress: {progress}%", 
                   (50, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw progress bar
        bar_width = 300
        filled_width = int(bar_width * progress / 100)
        cv2.rectangle(frame, (50, h - 30), (50 + bar_width, h - 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (50, h - 30), (50 + filled_width, h - 20), (0, 255, 255), -1)
        
        return frame
    
    def draw_measurement_screen(self, frame, measurements):
        """Draw measurement screen UI"""
        return self.draw_measurements(frame, measurements)
    
    def draw_input_prompt(self, frame):
        """Draw the input prompt UI for height and gender"""
        h, w = frame.shape[:2]
        panel_w = 400
        panel_h = 150
        panel_x = (w - panel_w) // 2
        panel_y = (h - panel_h) // 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Prompt message
        if self.input_type == "height":
            prompt = "Enter person's height in cm:"
        elif self.input_type == "gender":
            prompt = "Enter gender (m/f):"
        else:
            prompt = "Enter value:"
        
        cv2.putText(frame, prompt, (panel_x + 20, panel_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Input box
        input_box_x = panel_x + 20
        input_box_y = panel_y + 70
        cv2.rectangle(frame, (input_box_x, input_box_y - 30), (input_box_x + panel_w - 40, input_box_y + 10), (60, 60, 60), -1)
        cv2.putText(frame, self.input_text, (input_box_x + 10, input_box_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        cv2.putText(frame, "Press Enter to confirm", (panel_x + 20, panel_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    def process_frame(self, frame):
        """
        Process a single frame and return the processed frame.
        """
        if self.state == "SETUP":
            display_frame = self.draw_setup_screen(frame)
        elif self.state == "CALIBRATION":
            display_frame = self.draw_calibration_screen(frame)
            # Automatic calibration
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            if results and results.pose_landmarks:
                self.calibrate_with_multi_frame(frame)
                if self.calibration_complete:
                    self.state = "MEASUREMENT"
                    self.show_message("Calibration complete!", color=(0, 255, 0))
            
            # Draw pose landmarks during calibration
            if self.last_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    self.last_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            display_frame = frame # Use original frame to draw landmarks
        elif self.state == "MEASUREMENT":
            measurements = self.calculate_body_measurements(frame)

            if "error" in measurements:
                self.show_message(measurements["error"], color=(0, 0, 255))
                return self.draw_setup_screen(frame) # or return the original frame

            display_frame = self.draw_measurement_screen(frame, measurements)
        else:
            display_frame = frame
        
        # Show message
        if self.message and time.time() < self.message_timer:
            cv2.putText(display_frame, self.message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.message_color, 2)
        
        return display_frame

    def process_image(self, image_path):
        """Process an image file"""
        if not os.path.exists(image_path):
            self.show_message(f"Image file not found: {image_path}", color=(0, 0, 255))
            return None
            
        frame = cv2.imread(image_path)
        if frame is None:
            self.show_message(f"Could not read image: {image_path}", color=(0, 0, 255))
            return None
        
        # Simulate MEASUREMENT state and process the frame

        if not self.calibration_complete:
            if self.known_height_cm is None:
                self.show_message("Please set height for estimation or calibrate", color=(0, 0, 255))
                return frame
            else:
                #If height is set but calibration is not done, do some basic calibration
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb_frame)
                if results and results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        nose_x = landmarks[self.mp_pose.PoseLandmark.NOSE].x * w
                        nose_y = landmarks[self.mp_pose.PoseLandmark.NOSE].y * h

                        left_ankle_x = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].x * w
                        left_ankle_y = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE].y * h

                        right_ankle_x = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].x * w
                        right_ankle_y = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE].y * h

                        mid_ankle_x = (left_ankle_x + right_ankle_x) / 2
                        mid_ankle_y = (left_ankle_y + right_ankle_y) / 2
                        body_height_pixels = distance.euclidean((nose_x,nose_y), (mid_ankle_x, mid_ankle_y))
                        self.pixel_to_cm = (self.known_height_cm*0.95)/body_height_pixels
                        self.calibration_complete = True
                else:
                    self.show_message("Could not calibrate image due to pose detection failure", color=(0,0,255))
                    return frame


        measurements = self.calculate_body_measurements(frame)
        if "error" in measurements:
            self.show_message(measurements["error"], color=(0, 0, 255))
            return frame # Show the error on frame

        display_frame = self.draw_measurement_screen(frame, measurements)
        return display_frame

    def start_webcam(self):
        """Start capturing and processing from webcam with improved user interface"""
        # Try to open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            self.show_message("Could not open webcam", color=(0, 0, 255))
            return
        
        # Set a higher resolution if possible
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # FPS calculation
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        cv2.namedWindow("Body Measurement System", cv2.WINDOW_NORMAL)  # Create a named window
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                self.show_message("Could not read from webcam", color=(0, 0, 255))
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            if elapsed_time > 1:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
            
            # Process the frame
            display_frame = self.process_frame(frame)
            
            # Draw FPS
            cv2.putText(display_frame, f"FPS: {int(fps)}", (display_frame.shape[1] - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the frame
            cv2.imshow("Body Measurement System", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            self.handle_key_press(key)

            if key == ord('q'):
                try:
                    print("=========== Measurements ===========")
                    print(f"Chest Circumference: {self.measurement_history['chest'][-1]} cm")
                    print(f"Waist Circumference: {self.measurement_history['waist'][-1]} cm")
                    print(f"Height: {self.measurement_history['height'][-1]} cm")
                    print(f"Shoulder Width: {self.measurement_history['shoulder_width'][-1]} cm")
                    print(f"Size: {self.measurement_history['chest_size'][-1]}")
                    print(f"Waist Size: {self.measurement_history['waist_size'][-1]}")
                    print(f"FPS: {int(fps)}")
                    print("=====================================")
                except IndexError:
                    print("Exiting without measurements.")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nProgram terminated.")
    
    def handle_key_press(self, key):
        """Handles keyboard input for the webcam interface."""

        if self.input_active:
            if key == 13:  # Enter key
                self.input_active = False
                if self.input_type == "height":
                    self.set_known_height(self.input_text)
                elif self.input_type == "gender":
                    self.set_gender(self.input_text)
                self.input_text = ""
                self.input_type = None
            elif key == 8:  # Backspace
                self.input_text = self.input_text[:-1]
            elif key >= 32 and key <= 126:  # Printable characters
                self.input_text += chr(key)
        else:
            if key == ord('h'):
                self.input_active = True
                self.input_type = "height"
                self.show_message("Enter height in cm and press Enter", color=(255, 255, 0))

            elif key == ord('g'):
                self.input_active = True
                self.input_type = "gender"
                self.show_message("Enter gender (m/f) and press Enter", color=(255, 255, 0))

            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                self.show_message(f"Debug mode {'enabled' if self.debug_mode else 'disabled'}", color=(255, 255, 0))

            elif key == ord('r'):
                self.reset_calibration()
                self.state = "SETUP"
                self.show_message("Calibration reset", color=(255, 255, 0))

            elif key == ord('c') and self.state == "SETUP":
                self.state = "CALIBRATION"
                self.calibration_frames = []
                self.show_message("Entering Calibration...", color=(255, 255, 0))
            elif key == ord('s') and self.state == "CALIBRATION" and self.calibration_complete: # Use 's' to skip calibration
                self.state = "MEASUREMENT"
                self.show_message("Skipped Calibration. Using existing calibration data.", color=(255, 255, 0))
            elif key == ord('m') and self.state == "CALIBRATION" and self.calibration_complete:
                 self.state = "MEASUREMENT"
                 self.show_message("Measurement started", color=(255, 255, 0))

# Main program
def main():
    print("===== Body Measurement System =====")
    print("This program will measure chest, waist, and other body dimensions.")

    # Create measurement system
    system = BodyMeasurementSystem()
    
    # Choose mode
    print("\nSelect operation mode:")
    print("1. Webcam (live measurement)")
    print("2. Image file (process a photo)")

    mode = input("Enter mode (1/2): ")
    if mode == '1':
        # Start webcam interface
        system.start_webcam()
    elif mode == '2':
      # Force set state to setup for processing images
        system.state = "SETUP"

        image_path = input("Enter the path to the image file: ")

        # Provide the opportunity to set height and gender before processing
        print("Before processing the image, you can set the height and gender for better accuracy.")
        set_height = input("Do you want to set the height? (y/n): ").lower()
        if set_height == 'y':
            height_str = input("Enter height in cm: ")
            try:
                height = float(height_str)
                system.set_known_height(height)  # Call set_known_height with user input
            except ValueError:
                print("Invalid height entered.")

        set_gender = input("Do you want to set the gender? (y/n): ").lower()
        if set_gender == 'y':
            gender_str = input("Enter gender (m/f): ").lower()
            system.set_gender(gender_str)

        # After setup, simulate that calibration has been done (Use default calibration if no camera available)

        if system.known_height_cm:  # If the user has provided known_height
             #Attempt Calibration from the single image if height is known.
             test_frame = cv2.imread(image_path)
             if test_frame is not None:
                #Run the auto calibration if the known height is set from a single image

                h, w, _ = test_frame.shape
                rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                results = system.pose.process(rgb_frame)
                if results and results.pose_landmarks:
                            #Image auto calib succeed
                            landmarks = results.pose_landmarks.landmark
                            nose_x = landmarks[system.mp_pose.PoseLandmark.NOSE].x * w
                            nose_y = landmarks[system.mp_pose.PoseLandmark.NOSE].y * h

                            left_ankle_x = landmarks[system.mp_pose.PoseLandmark.LEFT_ANKLE].x * w
                            left_ankle_y = landmarks[system.mp_pose.PoseLandmark.LEFT_ANKLE].y * h

                            right_ankle_x = landmarks[system.mp_pose.PoseLandmark.RIGHT_ANKLE].x * w
                            right_ankle_y = landmarks[system.mp_pose.PoseLandmark.RIGHT_ANKLE].y * h

                            mid_ankle_x = (left_ankle_x + right_ankle_x) / 2
                            mid_ankle_y = (left_ankle_y + right_ankle_y) / 2
                            body_height_pixels = distance.euclidean((nose_x,nose_y), (mid_ankle_x, mid_ankle_y))
                            system.pixel_to_cm = (system.known_height_cm*0.95)/body_height_pixels
                            system.calibration_complete = True
                            print(f"\nAuto calibration from image Successful. Pixel To CM: {system.pixel_to_cm}")

                else:
                    print(f"\nAuto calibration from image has failed because the people are not dected.")

             else:
                print("Could not load and auto-calibrate the image. Image is none. ")
        else:
            print("Please provide height. Height is needed for initial calibration.")


        result_frame = system.process_image(image_path)  # Process image *after* setup
          
        if result_frame is not None:
            cv2.imshow("Measurement Results", result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Invalid mode. Exiting.")

if __name__ == "__main__":
    main()