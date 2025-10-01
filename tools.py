from typing import Tuple
from langchain_core.tools import tool
import cv2
import numpy as np
import uuid
from io import StringIO
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import logging

logger = logging.getLogger(__name__)
persistent_vars = {}
analysis_cache = {}

@tool
def advanced_image_processor(
    thought: str,
    python_code: str,
    image_path: str,
    operation_type: str = "general"
) -> Tuple[str, dict]:
    """
    Advanced image processing tool with OpenCV, PIL, and matplotlib support.
    
    Args:
        thought (str): Reasoning behind the operation
        python_code (str): Python code for image processing
        image_path (str): Path to input image
        operation_type (str): Type of operation (filter, detection, analysis, etc.)
    
    Returns:
        Tuple[str, dict]: Execution output and updated state
    """
    logger.debug(f"Processing image at path: {image_path}")
    current_variables = {}

    if not os.path.exists(image_path):
        error_msg = f"Error: Image not found at {image_path}"
        logger.error(error_msg)
        return error_msg, {
            "intermediate_outputs": [{"thought": thought, "code": python_code, "output": error_msg}]
        }

    try:
        current_variables["image"] = cv2.imread(image_path)
        if current_variables["image"] is None:
            error_msg = f"Error: Could not load image from {image_path}"
            logger.error(error_msg)
            return error_msg, {
                "intermediate_outputs": [{"thought": thought, "code": python_code, "output": "Could not load image"}]
            }

        current_variables["pil_image"] = Image.open(image_path)
        current_variables["image_path"] = image_path

        old_stdout = sys.stdout
        sys.stdout = StringIO()

        exec_globals = globals().copy()
        exec_globals.update(persistent_vars)
        exec_globals.update(current_variables)
        exec_globals.update({
            "cv2": cv2,
            "np": np,
            "plt": plt,
            "patches": patches,
            "Image": Image,
            "ImageDraw": ImageDraw,
            "ImageFont": ImageFont,
            "output_images": [],
            "analysis_data": {},
            "detection_results": []
        })

        exec(python_code, exec_globals)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        persistent_vars.update({k: v for k, v in exec_globals.items() 
                              if k not in globals() and not k.startswith('__')})

        updated_state = {
            "intermediate_outputs": [{
                "thought": thought,
                "code": python_code,
                "output": output or "Operation completed successfully",
                "operation_type": operation_type
            }],
            "current_variables": {k: str(v) if not isinstance(v, (np.ndarray, Image.Image)) else f"<{type(v).__name__}>" 
                                for k, v in persistent_vars.items()},
            "analysis_results": exec_globals.get("analysis_data", {})
        }

        if "output_images" in exec_globals and exec_globals["output_images"]:
            output_image_paths = []
            for i, img in enumerate(exec_globals["output_images"]):
                image_filename = os.path.join("static", "outputs", f"{uuid.uuid4()}.png")
                os.makedirs(os.path.dirname(image_filename), exist_ok=True)
                
                if isinstance(img, np.ndarray):
                    cv2.imwrite(image_filename, img)
                elif isinstance(img, Image.Image):
                    img.save(image_filename)
                
                output_image_paths.append(image_filename)
            
            updated_state["output_image_paths"] = output_image_paths

        logger.debug(f"Image processing successful. Output paths: {updated_state['output_image_paths']}")
        return output or "Operation completed successfully", updated_state

    except Exception as e:
        sys.stdout = old_stdout
        error_msg = f"Error: {str(e)}"
        logger.error(error_msg)
        return error_msg, {
            "intermediate_outputs": [{
                "thought": thought,
                "code": python_code,
                "output": error_msg,
                "operation_type": operation_type
            }]
        }

@tool
def object_detection_tool(
    image_path: str,
    detection_type: str = "general",
    confidence_threshold: float = 0.5
) -> Tuple[str, dict]:
    """
    Object detection tool using OpenCV.
    
    Args:
        image_path (str): Path to input image
        detection_type (str): Type of object detection (general, face, custom)
        confidence_threshold (float): Minimum confidence threshold for detections
    
    Returns:
        Tuple[str, dict]: Detection results and updated state
    """
    try:
        if not os.path.exists(image_path):
            error_msg = f"Error: Image not found at {image_path}"
            logger.error(error_msg)
            return error_msg, {}
            
        image = cv2.imread(image_path)
        if image is None:
            error_msg = f"Error: Could not load image from {image_path}"
            logger.error(error_msg)
            return error_msg, {}
            
        detections = []
        output_images = []
        
        if detection_type == "face":
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if face_cascade.empty():
                raise Exception("Could not load face cascade classifier")
                
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            result_image = image.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                detections.append({
                    "type": "face",
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "confidence": 0.9
                })
            
            output_images.append(result_image)
            
        elif detection_type == "edge":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            output_images.append(edges_colored)
            
        elif detection_type == "contour":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            result_image = image.copy()
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
            output_images.append(result_image)
            
            detections = [{
                "type": "contour",
                "area": cv2.contourArea(contour),
                "perimeter": cv2.arcLength(contour, True)
            } for contour in contours]

        output_image_paths = []
        for img in output_images:
            image_filename = os.path.join("static", "outputs", f"{uuid.uuid4()}.png")
            os.makedirs(os.path.dirname(image_filename), exist_ok=True)
            cv2.imwrite(image_filename, img)
            output_image_paths.append(image_filename)
        
        result = f"Detected {len(detections)} objects of type '{detection_type}'"
        logger.debug(f"Object detection successful. Output paths: {output_image_paths}")
        return result, {
            "output_image_paths": output_image_paths,
            "analysis_results": {
                "detections": detections,
                "detection_type": detection_type,
                "total_detected": len(detections)
            }
        }
        
    except Exception as e:
        error_msg = f"Detection error: {str(e)}"
        logger.error(error_msg)
        return error_msg, {}

@tool
def image_analysis_tool(
    image_path: str,
    analysis_type: str = "basic"
) -> Tuple[str, dict]:
    """
    Image analysis tool using OpenCV and PIL.
    
    Args:
        image_path (str): Path to input image
        analysis_type (str): Type of analysis (basic, advanced)
    
    Returns:
        Tuple[str, dict]: Analysis results and updated state
    """
    try:
        if not os.path.exists(image_path):
            error_msg = f"Error: Image not found at {image_path}"
            logger.error(error_msg)
            return error_msg, {}
            
        image = cv2.imread(image_path)
        if image is None:
            error_msg = f"Error: Could not load image from {image_path}"
            logger.error(error_msg)
            return error_msg, {}
            
        pil_image = Image.open(image_path)
        
        analysis_results = {
            "dimensions": {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) == 3 else 1
            },
            "file_info": {
                "format": pil_image.format,
                "mode": pil_image.mode,
                "size_bytes": os.path.getsize(image_path)
            }
        }
        
        if analysis_type in ["basic", "all"]:
            analysis_results["statistics"] = {
                "mean_bgr": [float(np.mean(image[:,:,i])) for i in range(3)],
                "std_bgr": [float(np.std(image[:,:,i])) for i in range(3)],
                "min_bgr": [float(np.min(image[:,:,i])) for i in range(3)],
                "max_bgr": [float(np.max(image[:,:,i])) for i in range(3)]
            }
        
        if analysis_type in ["histogram", "all"]:
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.plot(hist_b, color='blue')
            plt.title('Blue Channel')
            plt.subplot(1, 3, 2)
            plt.plot(hist_g, color='green')
            plt.title('Green Channel')
            plt.subplot(1, 3, 3)
            plt.plot(hist_r, color='red')
            plt.title('Red Channel')
            
            hist_filename = os.path.join("static", "outputs", f"histogram_{uuid.uuid4()}.png")
            os.makedirs(os.path.dirname(hist_filename), exist_ok=True)
            plt.savefig(hist_filename)
            plt.close()
            
            analysis_results["histogram_plot"] = hist_filename
        
        if analysis_type in ["color", "all"]:
            pixels = image.reshape(-1, 3)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            analysis_results["dominant_colors"] = [
                [int(c) for c in color] for color in kmeans.cluster_centers_
            ]
        
        result_text = f"Analysis completed for {analysis_type} analysis"
        logger.debug(f"Image analysis successful. Histogram path: {analysis_results.get('histogram_plot')}")
        return result_text, {
            "analysis_results": analysis_results,
            "output_image_paths": [analysis_results.get("histogram_plot")] if "histogram_plot" in analysis_results else []
        }
        
    except Exception as e:
        error_msg = f"Analysis error: {str(e)}"
        logger.error(error_msg)
        return error_msg, {}

@tool
def image_enhancement_tool(
    image_path: str,
    enhancement_type: str = "auto",
    intensity: float = 1.0
) -> Tuple[str, dict]:
    """
    Image enhancement tool using OpenCV.
    
    Args:
        image_path (str): Path to input image
        enhancement_type (str): Type of enhancement (auto, brightness, contrast)
        intensity (float): Intensity of enhancement
    
    Returns:
        Tuple[str, dict]: Enhanced image and updated state
    """
    try:
        if not os.path.exists(image_path):
            error_msg = f"Error: Image not found at {image_path}"
            logger.error(error_msg)
            return error_msg, {}
            
        image = cv2.imread(image_path)
        if image is None:
            error_msg = f"Error: Could not load image from {image_path}"
            logger.error(error_msg)
            return error_msg, {}
            
        enhanced_images = []
        
        if enhancement_type == "auto":
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            enhanced_images.append(enhanced)
            
        elif enhancement_type == "brightness":
            enhanced = cv2.convertScaleAbs(image, alpha=1.0, beta=int(30 * intensity))
            enhanced_images.append(enhanced)
            
        elif enhancement_type == "contrast":
            enhanced = cv2.convertScaleAbs(image, alpha=1.0 + intensity * 0.5, beta=0)
            enhanced_images.append(enhanced)
            
        elif enhancement_type == "sharpen":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * intensity
            enhanced = cv2.filter2D(image, -1, kernel)
            enhanced_images.append(enhanced)
            
        elif enhancement_type == "denoise":
            enhanced = cv2.fastNlMeansDenoisingColored(image, None, 10*intensity, 10*intensity, 7, 21)
            enhanced_images.append(enhanced)
        
        output_image_paths = []
        for img in enhanced_images:
            image_filename = os.path.join("static", "outputs", f"enhanced_{uuid.uuid4()}.png")
            os.makedirs(os.path.dirname(image_filename), exist_ok=True)
            cv2.imwrite(image_filename, img)
            output_image_paths.append(image_filename)
        
        logger.debug(f"Image enhancement successful. Output paths: {output_image_paths}")
        return f"Applied {enhancement_type} enhancement with intensity {intensity}", {
            "output_image_paths": output_image_paths
        }
        
    except Exception as e:
        error_msg = f"Enhancement error: {str(e)}"
        logger.error(error_msg)
        return error_msg, {}