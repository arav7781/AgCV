from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
from base64 import b64encode
import logging
from graph import make_tool_graph
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)
tool_agent = make_tool_graph()

def safe_get_message_content(msg):
    if hasattr(msg, 'content'):
        return msg.content
    elif hasattr(msg, 'text'):
        return msg.text
    elif isinstance(msg, dict):
        return msg.get('content', str(msg))
    return str(msg)

def safe_get_message_role(msg):
    if hasattr(msg, 'role'):
        return msg.role
    elif hasattr(msg, '__class__'):
        class_name = msg.__class__.__name__.lower()
        if 'human' in class_name:
            return "user"
        elif 'ai' in class_name or 'assistant' in class_name:
            return "assistant"
        elif 'system' in class_name:
            return "system"
    elif isinstance(msg, dict):
        return msg.get('role', 'assistant')
    return "assistant"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    upload_dir = "static/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'png'
    filename = os.path.join(upload_dir, f"{uuid.uuid4()}.{file_extension}")
    
    try:
        file.save(filename)
        logger.debug(f"Image saved to {filename}")
        return jsonify({
            "image_path": filename,
            "message": "Image uploaded successfully"
        })
    except Exception as e:
        logger.error(f"Failed to save image: {str(e)}")
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '')
        image_path = data.get('image_path', '')
        
        if not user_message:
            return jsonify({"error": "Message is required"}), 400
        
        if not image_path or not os.path.exists(image_path):
            error_msg = f"Image file not found at {image_path}"
            logger.error(error_msg)
            return jsonify({"error": error_msg}), 400

        logger.debug(f"Processing chat request with image_path: {image_path}, message: {user_message}")
        
        initial_state = {
            "messages": [HumanMessage(content=user_message)],
            "input_data": [{
                "variable_name": "image",
                "data_type": "image",
                "data_path": image_path
            }],
            "current_variables": {},
            "intermediate_outputs": [],
            "output_image_paths": [],
            "analysis_results": {},
            "image_path": image_path
        }
        
        result = tool_agent.invoke(initial_state)
        
        response_data = {
            "message": "Processing completed",
            "intermediate_outputs": result.get("intermediate_outputs", []),
            "analysis_results": result.get("analysis_results", {}),
            "current_variables": result.get("current_variables", {}),
            "messages": [],
            "output_images": []
        }
        
        for msg in result.get("messages", []):
            try:
                role = safe_get_message_role(msg)
                content = safe_get_message_content(msg)
                response_data["messages"].append({
                    "role": role,
                    "content": content
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                response_data["messages"].append({
                    "role": "assistant",
                    "content": "Message processing completed"
                })
        
        if result.get("output_image_paths"):
            for img_path in result["output_image_paths"]:
                if img_path and os.path.exists(img_path):
                    try:
                        with open(img_path, "rb") as img_file:
                            encoded_image = b64encode(img_file.read()).decode('utf-8')
                            response_data["output_images"].append({
                                "path": img_path,
                                "data": f"data:image/png;base64,{encoded_image}"
                            })
                    except Exception as e:
                        logger.error(f"Error encoding image {img_path}: {e}")
            
            if response_data["output_images"]:
                response_data["primary_output"] = response_data["output_images"][0]["data"]
        
        logger.debug(f"Chat response: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "intermediate_outputs": [],
            "messages": [{"role": "assistant", "content": f"An error occurred: {str(e)}"}]
        }), 500

@app.route('/image/<path:filename>')
def serve_image(filename):
    try:
        return send_file(filename)
    except Exception as e:
        logger.error(f"Image not found: {str(e)}")
        return jsonify({"error": f"Image not found: {str(e)}"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Computer Vision Agent is running",
        "tools_available": [
            "advanced_image_processor",
            "object_detection_tool",
            "image_analysis_tool",
            "image_enhancement_tool"
        ]
    })

@app.route('/capabilities', methods=['GET'])
def get_capabilities():
    return jsonify({
        "image_processing": [
            "Grayscale conversion",
            "Blur and sharpening",
            "Edge detection",
            "Morphological operations",
            "Color space conversions",
            "Histogram equalization",
            "Custom filtering"
        ],
        "object_detection": [
            "Face detection",
            "Edge detection",
            "Contour detection",
            "Custom object detection"
        ],
        "image_analysis": [
            "Basic statistics",
            "Color histograms",
            "Dominant color extraction",
            "Image properties",
            "Texture analysis"
        ],
        "enhancement": [
            "Auto enhancement",
            "Brightness adjustment",
            "Contrast adjustment",
            "Sharpening",
            "Noise reduction"
        ]
    })

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("static/outputs", exist_ok=True)
    logger.info("🚀 Enhanced Computer Vision Agent starting...")
    app.run(debug=True, port=5001, host='0.0.0.0')