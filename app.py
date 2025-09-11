from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
from datetime import datetime
import pytesseract
from PIL import ImageFont, ImageDraw
import json

# Set tesseract path if needed (uncomment and set your path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Path to the rules configuration file
RULES_FILE = "artwork_rules.json"

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Default rules structure
DEFAULT_RULES = {
    "platform_requirements": {
        "spotify": {
            "min_size": 640,
            "max_size": 3000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": "Spotify",
            "enabled": True
        },
        "apple": {
            "min_size": 1400,
            "max_size": 3000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "dpi": 72,
            "color_space": "RGB",
            "name": "Apple Music",
            "enabled": True
        }
    },
    "general_requirements": {
        "no_watermarks": {
            "description": "No watermarks",
            "severity": "high",
            "enabled": True
        },
        "no_copyright_text": {
            "description": "No copyright text (©, ®, ™)",
            "severity": "high",
            "enabled": True
        },
        "no_website_urls": {
            "description": "No website URLs",
            "severity": "high",
            "enabled": True
        },
        "no_social_handles": {
            "description": "No social media handles",
            "severity": "high",
            "enabled": True
        },
        "no_explicit_content": {
            "description": "No explicit content",
            "severity": "high",
            "enabled": True
        },
        "no_blurry_images": {
            "description": "Not blurry",
            "severity": "medium",
            "enabled": True
        },
        "no_pixelated_images": {
            "description": "Not pixelated",
            "severity": "medium",
            "enabled": True
        }
    },
    "quality_thresholds": {
        "sharpness_min": 100,
        "brightness_min": 40,
        "brightness_max": 160,
        "contrast_min": 40,
        "compression_quality_min": 80,
        "edge_density_watermark_threshold": 0.1,
        "text_likelihood_threshold": 0.05,
        "pixelation_diff_threshold": 30
    },
    "text_detection": {
        "forbidden_terms": [
            "copyright", "©", "®", "™", "www.", ".com", ".net", ".org",
            "@", "instagram", "facebook", "twitter", "tiktok", "youtube"
        ],
        "min_text_length": 3,
        "max_text_density": 0.05
    },
    "last_updated": None
}


def load_rules():
    """Load rules from JSON file or create default if not exists"""
    if os.path.exists(RULES_FILE):
        try:
            with open(RULES_FILE, 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_RULES
    else:
        with open(RULES_FILE, 'w') as f:
            json.dump(DEFAULT_RULES, f, indent=4)
        return DEFAULT_RULES


def save_rules(rules):
    """Save rules to JSON file"""
    rules['last_updated'] = datetime.now().isoformat()
    with open(RULES_FILE, 'w') as f:
        json.dump(rules, f, indent=4)
    return rules


class AdvancedArtworkAnalyzer:
    def __init__(self):
        self.rules_file = RULES_FILE
        self._rules = load_rules()
        self._last_modified = self._get_file_mtime()

    def _get_file_mtime(self):
        return os.path.getmtime(self.rules_file) if os.path.exists(self.rules_file) else 0

    def get_rules(self):
        """Get rules, reloading if file has been modified"""
        if not os.path.exists(self.rules_file):
            return self._rules

        current_modified = self._get_file_mtime()
        if current_modified > self._last_modified:
            self._rules = load_rules()
            self._last_modified = current_modified

        return self._rules

    def reload_rules(self):
        """Force immediate reload of rules"""
        self._rules = load_rules()
        self._last_modified = self._get_file_mtime()

    def analyze_image(self, image_path):
        rules = self.get_rules()
        platform_requirements = rules['platform_requirements']
        general_requirements = rules['general_requirements']
        quality_thresholds = rules['quality_thresholds']
        text_detection = rules['text_detection']

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format = img.format.lower() if img.format else 'unknown'
                mode = img.mode
                file_size = os.path.getsize(image_path)

                # Get DPI information
                dpi = img.info.get('dpi', (72, 72))[0] if 'dpi' in img.info else 72

                # Basic analysis
                sharpness = self.calculate_sharpness(img)
                brightness = self.calculate_brightness(img)
                contrast = self.calculate_contrast(img)

                # Advanced analysis
                has_watermark = self.detect_watermarks(img, quality_thresholds)
                has_text = self.detect_text(img, text_detection, quality_thresholds)
                is_blurry = sharpness < quality_thresholds['sharpness_min']
                is_pixelated = self.check_pixelation(img, quality_thresholds)
                color_analysis = self.analyze_colors(img)
                has_transparency = self.has_transparency(img)
                compression_quality = self.estimate_compression_quality(img)

                return {
                    'width': width, 'height': height, 'format': format,
                    'mode': mode, 'size': file_size, 'aspect_ratio': float(width / height),
                    'sharpness': float(sharpness), 'brightness': float(brightness), 'contrast': float(contrast),
                    'dpi': float(dpi), 'has_watermark': bool(has_watermark), 'has_text': bool(has_text),
                    'is_blurry': bool(is_blurry), 'is_pixelated': bool(is_pixelated),
                    'color_analysis': color_analysis, 'has_transparency': bool(has_transparency),
                    'compression_quality': int(compression_quality)
                }
        except Exception as e:
            return {'error': str(e)}

    def calculate_sharpness(self, image):
        try:
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            img_array = np.array(gray_image)
            return float(cv2.Laplacian(img_array, cv2.CV_64F).var())
        except:
            return 0.0

    def calculate_brightness(self, image):
        try:
            if image.mode != 'HSV':
                hsv_image = image.convert('HSV')
            else:
                hsv_image = image
            h, s, v = hsv_image.split()
            hist = v.histogram()
            pixels = sum(hist)
            brightness = scale = len(hist)
            for index in range(scale):
                ratio = hist[index] / pixels
                brightness += ratio * (-scale + index)
            return float(brightness / scale)
        except:
            return 0.0

    def calculate_contrast(self, image):
        try:
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
            img_array = np.array(gray_image)
            return float(img_array.std())
        except:
            return 0.0

    def detect_watermarks(self, image, quality_thresholds):
        """Detect potential watermarks in the image"""
        try:
            # Convert to grayscale
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image

            # Enhance edges to detect watermarks
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)

            # Calculate edge density (watermarks often create dense edge patterns)
            edge_density = np.mean(edges_array > 50)

            # If edge density is high in certain areas, it might be a watermark
            return bool(edge_density > quality_thresholds['edge_density_watermark_threshold'])
        except:
            return False

    def detect_text(self, image, text_detection, quality_thresholds):
        """Detect text in the image using OCR"""
        try:
            # Use pytesseract to detect text
            text = pytesseract.image_to_string(image)

            # Check for common forbidden text patterns
            text_lower = text.lower()
            for pattern in text_detection['forbidden_terms']:
                if pattern in text_lower:
                    return True

            return bool(len(text.strip()) > text_detection['min_text_length'])
        except:
            # If pytesseract is not available, use a simpler approach
            try:
                # Convert to grayscale and enhance contrast
                if image.mode != 'L':
                    gray = image.convert('L')
                else:
                    gray = image

                # Use edge detection to find text-like regions
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edges_array = np.array(edges)

                # Text regions typically have high edge density
                text_likelihood = np.mean(edges_array > 100)
                return bool(text_likelihood > quality_thresholds['text_likelihood_threshold'])
            except:
                return False

    def check_pixelation(self, image, quality_thresholds):
        """Check if image is pixelated"""
        try:
            # Resize down and back up to detect pixelation
            small = image.resize((image.width // 10, image.height // 10), Image.NEAREST)
            enlarged = small.resize((image.width, image.height), Image.NEAREST)

            # Calculate difference between original and pixelated version
            diff = np.mean(np.abs(np.array(image) - np.array(enlarged)))
            return bool(diff > quality_thresholds['pixelation_diff_threshold'])
        except:
            return False

    def analyze_colors(self, image):
        """Analyze color distribution in the image"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image

            # Convert to numpy array
            img_array = np.array(rgb_image)

            # Calculate color statistics
            mean_color = np.mean(img_array, axis=(0, 1))
            std_color = np.std(img_array, axis=(0, 1))

            # Check if image is mostly black/white
            is_grayscale = np.allclose(std_color, [0, 0, 0], atol=10)

            return {
                'mean_rgb': [float(x) for x in mean_color.tolist()],
                'std_rgb': [float(x) for x in std_color.tolist()],
                'is_grayscale': str(bool(is_grayscale))
            }
        except:
            return {'mean_rgb': [0.0, 0.0, 0.0], 'std_rgb': [0.0, 0.0, 0.0], 'is_grayscale': 'False'}

    def has_transparency(self, image):
        """Check if image has transparency"""
        try:
            return bool(image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info))
        except:
            return False

    def estimate_compression_quality(self, image):
        """Estimate JPEG compression quality"""
        try:
            if image.format != 'JPEG':
                return 100  # Not a JPEG, assume high quality

            # Simple estimation based on file size vs dimensions
            width, height = image.size
            expected_size = width * height * 3  # Uncompressed RGB
            actual_size = len(image.tobytes())

            if actual_size == 0:
                return 100

            compression_ratio = actual_size / expected_size
            # Map ratio to quality (this is a rough estimation)
            quality = min(100, int(compression_ratio * 100 * 1.5))
            return max(10, quality)
        except:
            return 50

    def check_requirements(self, analysis, platform):
        rules = self.get_rules()
        platform_requirements = rules['platform_requirements']
        general_requirements = rules['general_requirements']
        quality_thresholds = rules['quality_thresholds']

        if platform not in platform_requirements or not platform_requirements[platform]['enabled']:
            return []

        requirements = platform_requirements[platform]
        results = []

        # Size check
        min_size_pass = analysis['width'] >= requirements['min_size'] and analysis['height'] >= requirements['min_size']
        results.append({
            'test': f"Minimum size ({requirements['min_size']}x{requirements['min_size']}px)",
            'status': 'pass' if min_size_pass else 'fail',
            'message': f"Current: {analysis['width']}x{analysis['height']}px",
            'value': f"{analysis['width']}x{analysis['height']}px",
            'required': f"{requirements['min_size']}x{requirements['min_size']}px min"
        })

        max_size_pass = analysis['width'] <= requirements['max_size'] and analysis['height'] <= requirements['max_size']
        results.append({
            'test': f"Maximum size ({requirements['max_size']}x{requirements['max_size']}px)",
            'status': 'pass' if max_size_pass else 'fail',
            'message': f"Current: {analysis['width']}x{analysis['height']}px",
            'value': f"{analysis['width']}x{analysis['height']}px",
            'required': f"{requirements['max_size']}x{requirements['max_size']}px max"
        })

        # Aspect ratio check
        aspect_ratio_diff = abs(analysis['aspect_ratio'] - requirements['aspect_ratio'])
        aspect_ratio_pass = aspect_ratio_diff < 0.01
        results.append({
            'test': 'Square aspect ratio (1:1)',
            'status': 'pass' if aspect_ratio_pass else 'fail',
            'message': f"Current ratio: {analysis['aspect_ratio']:.3f}",
            'value': f"{analysis['aspect_ratio']:.3f}:1",
            'required': '1:1'
        })

        # Format check
        format_pass = analysis['format'] in requirements['formats']
        results.append({
            'test': 'Supported format (JPEG/PNG)',
            'status': 'pass' if format_pass else 'fail',
            'message': f"Current format: {analysis['format'].upper()}",
            'value': analysis['format'].upper(),
            'required': 'JPEG, PNG'
        })

        # File size check
        file_size_pass = requirements['min_file_size'] <= analysis['size'] <= requirements['max_file_size']
        results.append({
            'test': 'File size (100KB - 10MB)',
            'status': 'pass' if file_size_pass else 'fail',
            'message': f"Current size: {self.format_file_size(analysis['size'])}",
            'value': self.format_file_size(analysis['size']),
            'required': '100KB - 10MB'
        })

        # DPI check
        dpi_pass = analysis['dpi'] >= requirements.get('dpi', 72)
        results.append({
            'test': f"DPI ({requirements.get('dpi', 72)}+ recommended)",
            'status': 'pass' if dpi_pass else 'warning',
            'message': f"Current DPI: {analysis['dpi']}",
            'value': f"{analysis['dpi']} DPI",
            'required': f"{requirements.get('dpi', 72)}+ DPI"
        })

        # Color space check
        color_space_pass = analysis['mode'] == requirements['color_space']
        results.append({
            'test': 'Color space (RGB)',
            'status': 'pass' if color_space_pass else 'warning',
            'message': f"Current: {analysis['mode']}",
            'value': analysis['mode'],
            'required': 'RGB'
        })

        # Quality checks
        sharpness_status = 'pass' if analysis['sharpness'] > quality_thresholds['sharpness_min'] else 'warning' if \
            analysis['sharpness'] > quality_thresholds['sharpness_min'] / 2 else 'fail'
        results.append({
            'test': 'Image sharpness',
            'status': sharpness_status,
            'message': f"Sharpness score: {analysis['sharpness']:.2f}",
            'value': f"{analysis['sharpness']:.2f}",
            'required': f">{quality_thresholds['sharpness_min']} (higher is better)"
        })

        brightness_status = 'pass' if quality_thresholds['brightness_min'] <= analysis['brightness'] <= \
                                      quality_thresholds['brightness_max'] else 'warning'
        results.append({
            'test': 'Brightness level',
            'status': brightness_status,
            'message': f"Brightness: {analysis['brightness']:.2f}",
            'value': f"{analysis['brightness']:.2f}",
            'required': f"{quality_thresholds['brightness_min']}-{quality_thresholds['brightness_max']} (optimal range)"
        })

        contrast_status = 'pass' if analysis['contrast'] > quality_thresholds['contrast_min'] else 'warning' if \
            analysis['contrast'] > quality_thresholds['contrast_min'] / 2 else 'fail'
        results.append({
            'test': 'Contrast level',
            'status': contrast_status,
            'message': f"Contrast: {analysis['contrast']:.2f}",
            'value': f"{analysis['contrast']:.2f}",
            'required': f">{quality_thresholds['contrast_min']} (higher is better)"
        })

        # General requirements
        for req_key, req_config in general_requirements.items():
            if not req_config['enabled']:
                continue

            if req_key == 'no_watermarks':
                status = 'pass' if not analysis['has_watermark'] else 'fail'
                message = 'Watermarks detected' if analysis['has_watermark'] else 'No watermarks found'
                value = 'No' if not analysis['has_watermark'] else 'Yes'
            elif req_key == 'no_copyright_text' or req_key == 'no_website_urls' or req_key == 'no_social_handles':
                status = 'pass' if not analysis['has_text'] else 'fail'
                message = 'Text content detected' if analysis['has_text'] else 'No text content found'
                value = 'No' if not analysis['has_text'] else 'Yes'
            elif req_key == 'no_blurry_images':
                status = 'pass' if not analysis['is_blurry'] else 'fail'
                message = 'Image appears blurry' if analysis['is_blurry'] else 'Image is sharp'
                value = 'No' if not analysis['is_blurry'] else 'Yes'
            elif req_key == 'no_pixelated_images':
                status = 'pass' if not analysis['is_pixelated'] else 'fail'
                message = 'Image appears pixelated' if analysis['is_pixelated'] else 'Image is not pixelated'
                value = 'No' if not analysis['is_pixelated'] else 'Yes'
            else:
                continue

            results.append({
                'test': req_config['description'],
                'status': status,
                'message': message,
                'value': value,
                'required': req_config['description']
            })

        # Compression quality check
        compression_status = 'pass' if analysis['compression_quality'] > quality_thresholds[
            'compression_quality_min'] else 'warning' if analysis['compression_quality'] > quality_thresholds[
            'compression_quality_min'] / 2 else 'fail'
        results.append({
            'test': 'Compression quality',
            'status': compression_status,
            'message': f"Estimated quality: {analysis['compression_quality']}",
            'value': f"{analysis['compression_quality']}/100",
            'required': f">{quality_thresholds['compression_quality_min']} (higher is better)"
        })

        return results

    def format_file_size(self, bytes):
        if bytes == 0:
            return "0 Bytes"
        units = ["Bytes", "KB", "MB", "GB"]
        i = int(np.floor(np.log(bytes) / np.log(1024)))
        return f"{bytes / (1024 ** i):.2f} {units[i]}"

    def generate_recommendations(self, analysis, platform_results):
        """Generate recommendations based on analysis results"""
        rules = self.get_rules()
        quality_thresholds = rules['quality_thresholds']

        recommendations = []

        # Size recommendations
        if analysis['width'] < 1400 or analysis['height'] < 1400:
            recommendations.append({
                'priority': 'high',
                'message': 'Increase resolution to at least 1400x1400px for optimal quality',
                'category': 'dimensions'
            })
        elif analysis['width'] < 3000 or analysis['height'] < 3000:
            recommendations.append({
                'priority': 'low',
                'message': 'Consider using 3000x3000px for the highest quality display',
                'category': 'dimensions'
            })

        # Aspect ratio recommendations
        if abs(analysis['aspect_ratio'] - 1.0) > 0.01:
            recommendations.append({
                'priority': 'high',
                'message': 'Crop image to perfect square (1:1 aspect ratio)',
                'category': 'composition'
            })

        # Format recommendations
        if analysis['format'] not in ['jpg', 'jpeg', 'png']:
            recommendations.append({
                'priority': 'high',
                'message': 'Convert to JPEG or PNG format for compatibility',
                'category': 'format'
            })

        # File size recommendations
        if analysis['size'] > 5 * 1024 * 1024:
            recommendations.append({
                'priority': 'medium',
                'message': 'Optimize file size for faster uploads without sacrificing quality',
                'category': 'optimization'
            })
        elif analysis['size'] < 100 * 1024:
            recommendations.append({
                'priority': 'high',
                'message': 'Increase file quality - current size is too small for good resolution',
                'category': 'quality'
            })

        # Quality recommendations
        if analysis['sharpness'] < quality_thresholds['sharpness_min']:
            recommendations.append({
                'priority': 'medium',
                'message': 'Increase image sharpness for better clarity',
                'category': 'quality'
            })

        if analysis['brightness'] < quality_thresholds['brightness_min']:
            recommendations.append({
                'priority': 'medium',
                'message': 'Increase image brightness for better visibility',
                'category': 'quality'
            })
        elif analysis['brightness'] > quality_thresholds['brightness_max']:
            recommendations.append({
                'priority': 'medium',
                'message': 'Reduce image brightness to avoid washed-out appearance',
                'category': 'quality'
            })

        if analysis['contrast'] < quality_thresholds['contrast_min']:
            recommendations.append({
                'priority': 'medium',
                'message': 'Increase contrast for better visual impact',
                'category': 'quality'
            })

        # Watermark and text recommendations
        if analysis['has_watermark']:
            recommendations.append({
                'priority': 'high',
                'message': 'Remove any watermarks from the artwork',
                'category': 'content'
            })

        if analysis['has_text']:
            recommendations.append({
                'priority': 'high',
                'message': 'Remove any text, URLs, or social media handles from the artwork',
                'category': 'content'
            })

        # Compression recommendations
        if analysis.get('compression_quality', 100) < quality_thresholds['compression_quality_min']:
            recommendations.append({
                'priority': 'medium',
                'message': f"Use higher quality compression settings ({quality_thresholds['compression_quality_min']}% or higher)",
                'category': 'quality'
            })

        return recommendations

    def determine_overall_status(self, platform_results):
        """Determine overall approval status"""
        # Count passes, warnings, and failures
        all_statuses = [result['status'] for result in platform_results]

        if 'fail' in all_statuses:
            return 'fail'
        elif 'warning' in all_statuses:
            return 'warning'
        else:
            return 'pass'


analyzer = AdvancedArtworkAnalyzer()


@app.route('/')
def index():
    return render_template('index.html')


def make_serializable(obj):
    """Convert non-serializable objects to serializable formats"""
    if isinstance(obj, (bool, np.bool_)):
        return str(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    else:
        return obj


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
        file.save(filepath)

        # Analyze the image
        analysis = analyzer.analyze_image(filepath)
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 500

        # Check requirements for all platforms
        platform_results = {}
        recommendations = {}
        overall_statuses = {}

        rules = analyzer.get_rules()
        for platform, config in rules['platform_requirements'].items():
            # Check requirements for all platforms
            platform_results[platform] = analyzer.check_requirements(analysis, platform)

            # Only generate recommendations for enabled platforms
            if config['enabled']:
                recommendations[platform] = analyzer.generate_recommendations(analysis, platform_results[platform])
                overall_statuses[platform] = analyzer.determine_overall_status(platform_results[platform])
            else:
                recommendations[platform] = []
                overall_statuses[platform] = 'disabled'

        # Prepare response
        response = {
            'analysis': make_serializable(analysis),
            'platform_results': make_serializable(platform_results),
            'recommendations': make_serializable(recommendations),
            'overall_statuses': overall_statuses,
            'platform_configs': {platform: {'enabled': config['enabled'], 'name': config['name']}
                                 for platform, config in rules['platform_requirements'].items()},
            'specifications': {
                'dimensions': f"{analysis['width']} × {analysis['height']} pixels",
                'file_size': analyzer.format_file_size(analysis['size']),
                'format': analysis['format'].upper(),
                'aspect_ratio': f"{analysis['aspect_ratio']:.3f}:1",
                'color_mode': analysis['mode'],
                'dpi': f"{analysis['dpi']} DPI",
                'sharpness': f"{analysis['sharpness']:.2f}",
                'brightness': f"{analysis['brightness']:.2f}",
                'contrast': f"{analysis['contrast']:.2f}",
                'compression_quality': analysis.get('compression_quality', 'N/A')
            }
        }

        return jsonify(response)


# Admin routes
@app.route('/admin')
def admin_dashboard():
    """Admin dashboard"""
    rules = load_rules()
    return render_template('admin_dashboard.html', rules=rules)


@app.route('/admin/rules/platform/<platform>', methods=['GET', 'POST'])
def edit_platform_rules(platform):
    """Edit platform-specific rules"""
    rules = load_rules()

    if platform not in rules['platform_requirements']:
        return "Platform not found", 404

    if request.method == 'POST':
        # Update platform rules
        platform_rules = rules['platform_requirements'][platform]

        # Update numeric values
        platform_rules['min_size'] = int(request.form.get('min_size', platform_rules['min_size']))
        platform_rules['max_size'] = int(request.form.get('max_size', platform_rules['max_size']))
        platform_rules['aspect_ratio'] = float(request.form.get('aspect_ratio', platform_rules['aspect_ratio']))
        platform_rules['max_file_size'] = int(request.form.get('max_file_size', platform_rules['max_file_size']))
        platform_rules['min_file_size'] = int(request.form.get('min_file_size', platform_rules['min_file_size']))
        platform_rules['dpi'] = int(request.form.get('dpi', platform_rules['dpi']))

        # Update formats
        formats = request.form.get('formats', '').split(',')
        platform_rules['formats'] = [f.strip().lower() for f in formats if f.strip()]

        # Update color space
        platform_rules['color_space'] = request.form.get('color_space', platform_rules['color_space'])

        # Update display name
        platform_rules['name'] = request.form.get('name', platform_rules['name'])

        # Update enabled status
        platform_rules['enabled'] = request.form.get('enabled') == 'on'

        # Save updated rules
        save_rules(rules)
        analyzer.reload_rules()  # Force reload

        return redirect(url_for('admin_dashboard'))

    return render_template('edit_platform_rules.html',
                           platform=platform,
                           rules=rules['platform_requirements'][platform])


@app.route('/admin/rules/general', methods=['GET', 'POST'])
def edit_general_rules():
    """Edit general rules"""
    rules = load_rules()

    if request.method == 'POST':
        # Update general rules
        for rule_key in rules['general_requirements']:
            enabled = request.form.get(f"{rule_key}_enabled") == "on"
            rules['general_requirements'][rule_key]['enabled'] = enabled

            # Update description if provided
            new_desc = request.form.get(f"{rule_key}_description")
            if new_desc:
                rules['general_requirements'][rule_key]['description'] = new_desc

            # Update severity if provided
            new_severity = request.form.get(f"{rule_key}_severity")
            if new_severity in ['high', 'medium', 'low']:
                rules['general_requirements'][rule_key]['severity'] = new_severity

        # Save updated rules
        save_rules(rules)
        analyzer.reload_rules()  # Force reload

        return redirect(url_for('admin_dashboard'))

    return render_template('edit_general_rules.html', rules=rules['general_requirements'])


@app.route('/admin/rules/quality', methods=['GET', 'POST'])
def edit_quality_rules():
    """Edit quality thresholds"""
    rules = load_rules()

    if request.method == 'POST':
        # Update quality thresholds
        thresholds = rules['quality_thresholds']

        thresholds['sharpness_min'] = float(request.form.get('sharpness_min', thresholds['sharpness_min']))
        thresholds['brightness_min'] = float(request.form.get('brightness_min', thresholds['brightness_min']))
        thresholds['brightness_max'] = float(request.form.get('brightness_max', thresholds['brightness_max']))
        thresholds['contrast_min'] = float(request.form.get('contrast_min', thresholds['contrast_min']))
        thresholds['compression_quality_min'] = float(
            request.form.get('compression_quality_min', thresholds['compression_quality_min']))
        thresholds['edge_density_watermark_threshold'] = float(
            request.form.get('edge_density_watermark_threshold', thresholds['edge_density_watermark_threshold']))
        thresholds['text_likelihood_threshold'] = float(
            request.form.get('text_likelihood_threshold', thresholds['text_likelihood_threshold']))
        thresholds['pixelation_diff_threshold'] = float(
            request.form.get('pixelation_diff_threshold', thresholds['pixelation_diff_threshold']))

        # Save updated rules
        save_rules(rules)
        analyzer.reload_rules()  # Force reload

        return redirect(url_for('admin_dashboard'))

    return render_template('edit_quality_rules.html', thresholds=rules['quality_thresholds'])


@app.route('/admin/rules/text', methods=['GET', 'POST'])
def edit_text_rules():
    """Edit text detection rules"""
    rules = load_rules()

    if request.method == 'POST':
        # Update text detection rules
        text_rules = rules['text_detection']

        # Update forbidden terms
        terms = request.form.get('forbidden_terms', '')
        text_rules['forbidden_terms'] = [term.strip().lower() for term in terms.split(',') if term.strip()]

        # Update other text settings
        text_rules['min_text_length'] = int(request.form.get('min_text_length', text_rules['min_text_length']))
        text_rules['max_text_density'] = float(request.form.get('max_text_density', text_rules['max_text_density']))

        # Save updated rules
        save_rules(rules)
        analyzer.reload_rules()  # Force reload

        return redirect(url_for('admin_dashboard'))

    return render_template('edit_text_rules.html', text_rules=rules['text_detection'])


@app.route('/admin/rules/reset', methods=['POST'])
def reset_rules():
    """Reset all rules to defaults"""
    save_rules(DEFAULT_RULES)
    analyzer.reload_rules()  # Force reload
    return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    # Ensure rules file exists
    load_rules()

    app.run(debug=True, port=5000)