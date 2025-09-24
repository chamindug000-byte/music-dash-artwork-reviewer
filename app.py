from flask import Flask, render_template, request, jsonify, redirect, url_for
from sympy import false
from werkzeug.utils import secure_filename
import os
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import numpy as np
import cv2
from datetime import datetime
import pytesseract
from PIL import ImageFont, ImageDraw
import json
import requests
from io import BytesIO
import re
from sklearn.cluster import DBSCAN
import easyocr
import time
from deep_translator import GoogleTranslator
import concurrent.futures

# Set tesseract path if needed (uncomment and set your path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Path to the rules configuration file
RULES_FILE = "artwork_rules.json"

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize EasyOCR reader (once at startup for better performance)
try:
    reader = easyocr.Reader(['en'])  # Initialize English reader
    EASY_OCR_AVAILABLE = True
except:
    EASY_OCR_AVAILABLE = False
    print("EasyOCR not available, falling back to Tesseract")

# Bad words list (comprehensive list)
BAD_WORDS = [
    "fuck", "shit", "asshole", "bitch", "cunt", "dick", "piss", "cock", "pussy", "whore",
    "slut", "bastard", "nigga", "nigger", "faggot", "kys", "kill yourself", "retard",
    "cocksucker", "motherfucker", "douchebag", "scumbag", "shithead", "dickhead", "prick",
    "arsehole", "bollocks", "wanker", "twat", "bloody", "bugger", "crap", "damn", "hell",
    "fag", "dyke", "tranny", "whore", "hooker", "prostitute", "pedophile", "rapist"
]

# Weapon-related terms for text detection
WEAPON_WORDS = [
    "gun", "pistol", "rifle", "shotgun", "ak47", "ak-47", "uzi", "machine gun", "revolver",
    "knife", "dagger", "sword", "machete", "bomb", "grenade", "explosive", "firearm",
    "weapon", "ammo", "bullet", "cartridge", "arsenal", "kill", "murder", "shoot"
]

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
            "enabled": True,
            "approval_threshold": 80  # 80% score required for approval
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
            "enabled": True,
            "approval_threshold": 80
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
        "no_blurry_images": {
            "description": "Not blurry",
            "severity": "medium",
            "enabled": True
        },
        "no_pixelated_images": {
            "description": "Not pixelated",
            "severity": "medium",
            "enabled": False
        },
        "no_logos": {
            "description": "No logos or branding elements",
            "severity": "high",
            "enabled": True
        },
        "no_bad_words": {
            "description": "No offensive or inappropriate language",
            "severity": "high",
            "enabled": True
        },
        "no_weapons": {
            "description": "No weapons or violent content",
            "severity": "high",
            "enabled": True
        },
        "has_artist_info": {
            "description": "Contains artist/album/song information",
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
        "pixelation_diff_threshold": 0,
        "logo_similarity_threshold": 0.7,
        "logo_contour_area_threshold": 0.01,
        "logo_correlation_threshold": 0.5,
        "text_region_min_aspect_ratio": 2.0,
        "text_region_max_aspect_ratio": 10.0,
        "logo_min_confidence": 0.6,
        "logo_max_size_ratio": 0.3,
        "weapon_confidence_threshold": 0.5,
        "artist_info_confidence": 0.7
    },
    "text_detection": {
        "forbidden_terms": [
            "copyright", "©", "®", "™", "www.", ".com", ".net", ".org",
            "@", "instagram", "facebook", "twitter", "tiktok", "youtube"
        ],
        "bad_words": BAD_WORDS,
        "weapon_words": WEAPON_WORDS,
        "artist_terms": [
            "artist", "album", "song", "track", "ep", "lp", "single", "feat", "ft",
            "presents", "mix", "remix", "records", "music", "sound", "audio",
            "producer", "dj", "band", "featuring", "presents", "presents"
        ],
        "allowed_terms": [
            "artist", "album", "song", "track", "ep", "lp", "single", "feat", "ft",
            "presents", "mix", "remix"
        ],
        "min_text_length": 3,
        "max_text_density": 0.05,
        "text_position_threshold": 0.1,
        "font_size_variation_threshold": 0.3
    },
    "performance_settings": {
        "max_image_size": 2000,
        "logo_detection_enabled": True,
        "text_detection_enabled": True,
        "watermark_detection_enabled": True,
        "weapon_detection_enabled": True,
        "fast_analysis_mode": False,
        "max_workers": 4,
        "optimize_speed": True
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
        self.logo_templates = self._load_logo_templates()
        self._logo_detector = None
        self.weapon_cascade = self._load_weapon_cascade()

    def _get_file_mtime(self):
        return os.path.getmtime(self.rules_file) if os.path.exists(self.rules_file) else 0

    def _load_logo_templates(self):
        """Load common logo templates for detection"""
        templates = {}
        return templates

    def _load_weapon_cascade(self):
        """Load weapon detection cascade classifier"""
        try:
            return None
        except:
            return None

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
        performance_settings = rules.get('performance_settings', {})

        start_time = time.time()

        try:
            with Image.open(image_path) as img:
                # Resize image if too large for faster processing
                max_size = performance_settings.get('max_image_size', 2000)
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.LANCZOS)

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

                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(
                        max_workers=performance_settings.get('max_workers', 4)
                ) as executor:
                    # Submit analysis tasks
                    watermark_future = executor.submit(
                        self.detect_watermarks, img, rules['quality_thresholds']
                    ) if performance_settings.get('watermark_detection_enabled', True) else None

                    text_future = executor.submit(
                        self.detect_text, img, rules['text_detection'], rules['quality_thresholds']
                    ) if performance_settings.get('text_detection_enabled', True) else None

                    logo_future = executor.submit(
                        self.detect_logos, img, rules['quality_thresholds']
                    ) if performance_settings.get('logo_detection_enabled', True) else None

                    weapon_future = executor.submit(
                        self.detect_weapons, img, rules['quality_thresholds']
                    ) if performance_settings.get('weapon_detection_enabled', True) else None

                    # Get results
                    has_watermark = watermark_future.result() if watermark_future else False
                    text_results = text_future.result() if text_future else {
                        'has_text': False, 'detected_text': [], 'forbidden_text_found': False,
                        'bad_words_found': False, 'weapon_words_found': False, 'artist_info_found': False,
                        'text_analysis': {}
                    }
                    logo_results = logo_future.result() if logo_future else {
                        'has_logos': False, 'logo_count': 0, 'logo_locations': []
                    }
                    weapon_results = weapon_future.result() if weapon_future else {
                        'has_weapons': False, 'weapon_confidence': 0, 'weapon_details': []
                    }

                # Other analyses
                is_blurry = sharpness < rules['quality_thresholds']['sharpness_min']
                is_pixelated = self.check_pixelation(img, rules['quality_thresholds'])
                color_analysis = self.analyze_colors(img)
                has_transparency = self.has_transparency(img)
                compression_quality = self.estimate_compression_quality(img)

                processing_time = time.time() - start_time

                return {
                    'width': width, 'height': height, 'format': format,
                    'mode': mode, 'size': file_size, 'aspect_ratio': float(width / height),
                    'sharpness': float(sharpness), 'brightness': float(brightness), 'contrast': float(contrast),
                    'dpi': float(dpi), 'has_watermark': bool(has_watermark),
                    'has_text': bool(text_results['has_text']), 'detected_text': text_results['detected_text'],
                    'is_blurry': bool(is_blurry), 'is_pixelated': bool(is_pixelated),
                    'color_analysis': color_analysis, 'has_transparency': bool(has_transparency),
                    'compression_quality': int(compression_quality),
                    'has_logos': bool(logo_results['has_logos']), 'logo_count': int(logo_results['logo_count']),
                    'logo_locations': logo_results.get('logo_locations', []),
                    'forbidden_text_found': text_results.get('forbidden_text_found', False),
                    'bad_words_found': text_results.get('bad_words_found', False),
                    'weapon_words_found': text_results.get('weapon_words_found', False),
                    'artist_info_found': text_results.get('artist_info_found', False),
                    'text_analysis': text_results.get('text_analysis', {}),
                    'has_weapons': bool(weapon_results.get('has_weapons', False)),
                    'weapon_confidence': float(weapon_results.get('weapon_confidence', 0)),
                    'weapon_details': weapon_results.get('weapon_details', []),
                    'processing_time': float(processing_time)
                }
        except Exception as e:
            return {'error': str(e)}

    def calculate_sharpness(self, image):
        try:
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image

            # Resize for faster processing if needed
            if gray_image.width > 800 or gray_image.height > 800:
                gray_image.thumbnail((800, 800), Image.LANCZOS)

            img_array = np.array(gray_image)
            return float(cv2.Laplacian(img_array, cv2.CV_64F).var())
        except:
            return 0.0

    def calculate_brightness(self, image):
        try:
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image

            # Resize for faster processing if needed
            if gray_image.width > 500 or gray_image.height > 500:
                gray_image.thumbnail((500, 500), Image.LANCZOS)

            img_array = np.array(gray_image)
            return float(np.mean(img_array))
        except:
            return 0.0

    def calculate_contrast(self, image):
        try:
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image

            # Resize for faster processing if needed
            if gray_image.width > 500 or gray_image.height > 500:
                gray_image.thumbnail((500, 500), Image.LANCZOS)

            img_array = np.array(gray_image)
            return float(img_array.std())
        except:
            return 0.0

    def detect_watermarks(self, image, quality_thresholds):
        """Detect potential watermarks in the image"""
        try:
            if image.mode != 'L':
                gray = image.convert('L')
            else:
                gray = image

            if gray.width > 800 or gray.height > 800:
                gray.thumbnail((800, 800), Image.LANCZOS)

            edges = gray.filter(ImageFilter.FIND_EDGES)
            edges_array = np.array(edges)

            edge_density = np.mean(edges_array > 50)
            return bool(edge_density > quality_thresholds['edge_density_watermark_threshold'])
        except:
            return False

    def detect_text(self, image, text_detection, quality_thresholds):
        """Detect text in the image using OCR with improved accuracy"""
        result = {
            'has_text': False,
            'detected_text': [],
            'text_confidence': 0,
            'forbidden_text_found': False,
            'bad_words_found': False,
            'weapon_words_found': False,
            'artist_info_found': False,
            'text_analysis': {
                'is_artistic': False,
                'text_locations': [],
                'font_variation': 0,
                'central_text_ratio': 0
            }
        }

        try:
            processed_img = self.preprocess_image_for_ocr(image)
            width, height = processed_img.size

            if EASY_OCR_AVAILABLE:
                img_array = np.array(processed_img.convert('RGB'))
                text_data = reader.readtext(img_array)

                detected_texts = []
                forbidden_found = False
                bad_words_found = False
                weapon_words_found = False
                artist_info_found = False
                total_confidence = 0
                valid_text_count = 0
                text_locations = []

                for detection in text_data:
                    bbox, text, confidence = detection

                    if (confidence > 0.6 and len(text.strip()) >= text_detection['min_text_length'] and
                            not text.strip().isspace() and text.strip() != ''):

                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        bbox_simple = (
                            min(x_coords), min(y_coords),
                            max(x_coords) - min(x_coords),
                            max(y_coords) - min(y_coords)
                        )

                        detected_texts.append({
                            'text': text.strip(),
                            'confidence': confidence * 100,
                            'bounding_box': bbox_simple
                        })

                        text_locations.append({
                            'x': bbox_simple[0],
                            'y': bbox_simple[1],
                            'width': bbox_simple[2],
                            'height': bbox_simple[3],
                            'text': text.strip()
                        })

                        total_confidence += confidence * 100
                        valid_text_count += 1

                        # Check for different types of text
                        text_lower = text.lower()

                        # Check forbidden terms
                        for pattern in text_detection['forbidden_terms']:
                            if pattern in text_lower:
                                forbidden_found = True
                                break

                        # Check bad words
                        for bad_word in text_detection.get('bad_words', BAD_WORDS):
                            if bad_word in text_lower:
                                bad_words_found = True
                                break

                        # Check weapon words
                        for weapon_word in text_detection.get('weapon_words', WEAPON_WORDS):
                            if weapon_word in text_lower:
                                weapon_words_found = True
                                break

                        # Check artist info
                        for artist_term in text_detection.get('artist_terms', []):
                            if artist_term in text_lower:
                                artist_info_found = True
                                break

            else:
                # Fallback to Tesseract
                custom_config = r'--oem 3 --psm 6'
                text_data = pytesseract.image_to_data(processed_img, output_type=pytesseract.Output.DICT,
                                                      config=custom_config)

                detected_texts = []
                forbidden_found = False
                bad_words_found = False
                weapon_words_found = False
                artist_info_found = False
                total_confidence = 0
                valid_text_count = 0
                text_locations = []

                for i in range(len(text_data['text'])):
                    text = text_data['text'][i].strip()
                    confidence = int(text_data['conf'][i])

                    if (confidence > 60 and len(text) >= text_detection['min_text_length'] and
                            not text.isspace() and text != ''):

                        detected_texts.append({
                            'text': text,
                            'confidence': confidence,
                            'bounding_box': (
                                text_data['left'][i],
                                text_data['top'][i],
                                text_data['width'][i],
                                text_data['height'][i]
                            )
                        })

                        text_locations.append({
                            'x': text_data['left'][i],
                            'y': text_data['top'][i],
                            'width': text_data['width'][i],
                            'height': text_data['height'][i],
                            'text': text
                        })

                        total_confidence += confidence
                        valid_text_count += 1

                        text_lower = text.lower()

                        # Check forbidden terms
                        for pattern in text_detection['forbidden_terms']:
                            if pattern in text_lower:
                                forbidden_found = True
                                break

                        # Check bad words
                        for bad_word in text_detection.get('bad_words', BAD_WORDS):
                            if bad_word in text_lower:
                                bad_words_found = True
                                break

                        # Check weapon words
                        for weapon_word in text_detection.get('weapon_words', WEAPON_WORDS):
                            if weapon_word in text_lower:
                                weapon_words_found = True
                                break

                        # Check artist info
                        for artist_term in text_detection.get('artist_terms', []):
                            if artist_term in text_lower:
                                artist_info_found = True
                                break

            if valid_text_count > 0:
                result['has_text'] = True
                result['detected_text'] = detected_texts
                result['text_confidence'] = total_confidence / valid_text_count
                result['forbidden_text_found'] = forbidden_found
                result['bad_words_found'] = bad_words_found
                result['weapon_words_found'] = weapon_words_found
                result['artist_info_found'] = artist_info_found

                # Analyze text characteristics
                text_analysis = self.analyze_text_characteristics(text_locations, width, height)
                result['text_analysis'] = text_analysis

            return result

        except Exception as e:
            print(f"Text detection error: {e}")
            try:
                if image.mode != 'L':
                    gray = image.convert('L')
                else:
                    gray = image

                edges = gray.filter(ImageFilter.FIND_EDGES)
                edges_array = np.array(edges)
                text_likelihood = np.mean(edges_array > 100)
                result['has_text'] = bool(text_likelihood > quality_thresholds['text_likelihood_threshold'])
                return result
            except:
                return result

    def analyze_text_characteristics(self, text_locations, image_width, image_height):
        """Analyze text to determine if it's artistic or informational"""
        if not text_locations:
            return {
                'is_artistic': False,
                'text_locations': [],
                'font_variation': 0,
                'central_text_ratio': 0
            }

        central_text_count = 0
        font_sizes = []

        for location in text_locations:
            x_center = location['x'] + location['width'] / 2
            y_center = location['y'] + location['height'] / 2

            center_threshold = self.get_rules()['text_detection'].get('text_position_threshold', 0.1)
            if (center_threshold * image_width <= x_center <= (1 - center_threshold) * image_width and
                    center_threshold * image_height <= y_center <= (1 - center_threshold) * image_height):
                central_text_count += 1

            font_sizes.append(location['height'])

        if font_sizes:
            font_variation = np.std(font_sizes) / np.mean(font_sizes) if np.mean(font_sizes) > 0 else 0
        else:
            font_variation = 0

        central_ratio = central_text_count / len(text_locations) if text_locations else 0
        variation_threshold = self.get_rules()['text_detection'].get('font_size_variation_threshold', 0.3)

        is_artistic = (central_ratio > 0.7 and font_variation > variation_threshold)

        return {
            'is_artistic': is_artistic,
            'text_locations': text_locations,
            'font_variation': font_variation,
            'central_text_ratio': central_ratio
        }

    def preprocess_image_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy"""
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image

        if gray.width > 800 or gray.height > 800:
            gray.thumbnail((800, 800), Image.LANCZOS)

        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)

        sharpened = enhanced.filter(ImageFilter.SHARPEN)

        width, height = sharpened.size
        if width < 400 or height < 400:
            sharpened = sharpened.resize((width * 2, height * 2), Image.LANCZOS)

        return sharpened

    def detect_logos(self, image, quality_thresholds):
        """Improved logo detection with better accuracy"""
        result = {
            'has_logos': False,
            'logo_count': 0,
            'logo_locations': [],
            'logo_confidence': 0
        }

        try:
            if image.mode == 'RGBA':
                image_rgb = Image.new('RGB', image.size, (255, 255, 255))
                image_rgb.paste(image, mask=image.split()[3])
                img_array = np.array(image_rgb)
            else:
                img_array = np.array(image.convert('RGB'))

            # Resize for faster processing
            if img_array.shape[1] > 600 or img_array.shape[0] > 600:
                scale_factor = min(600 / img_array.shape[1], 600 / img_array.shape[0])
                new_width = int(img_array.shape[1] * scale_factor)
                new_height = int(img_array.shape[0] * scale_factor)
                img_array = cv2.resize(img_array, (new_width, new_height))

            # Use multiple detection methods for better accuracy
            contour_logos = self.detect_logos_by_contour(img_array, quality_thresholds)
            feature_logos = self.detect_logos_by_features(img_array, quality_thresholds)
            template_logos = self.detect_logos_by_template(img_array, quality_thresholds)

            all_logos = contour_logos + feature_logos + template_logos

            # Remove duplicates and low-confidence detections
            if all_logos:
                # Filter by confidence
                min_confidence = quality_thresholds.get('logo_min_confidence', 0.6)
                filtered_logos = [logo for logo in all_logos if logo.get('confidence', 0) >= min_confidence]

                # Cluster nearby detections
                if len(filtered_logos) > 1:
                    logo_points = np.array([[(logo['x'] + logo['width'] / 2),
                                             (logo['y'] + logo['height'] / 2)] for logo in filtered_logos])

                    clustering = DBSCAN(eps=25, min_samples=1).fit(logo_points)
                    labels = clustering.labels_

                    clusters = {}
                    for i, label in enumerate(labels):
                        if label not in clusters:
                            clusters[label] = []
                        clusters[label].append(filtered_logos[i])

                    unique_logos = []
                    for cluster in clusters.values():
                        cluster.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                        unique_logos.append(cluster[0])

                    result['logo_count'] = len(unique_logos)
                    result['logo_locations'] = unique_logos
                    if unique_logos:
                        result['logo_confidence'] = max([logo.get('confidence', 0) for logo in unique_logos])
                else:
                    if filtered_logos:
                        result['logo_count'] = 1
                        result['logo_locations'] = filtered_logos
                        result['logo_confidence'] = filtered_logos[0].get('confidence', 0)

            result['has_logos'] = result['logo_count'] > 0
            return result

        except Exception as e:
            print(f"Logo detection error: {e}")
            return result

    def detect_logos_by_contour(self, img_array, quality_thresholds):
        """Detect logos using contour analysis"""
        logos = []

        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                total_area = gray.shape[0] * gray.shape[1]
                contour_threshold = quality_thresholds.get('logo_contour_area_threshold', 0.01)

                if area / total_area > contour_threshold:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h

                    if 0.3 < aspect_ratio < 3.0:  # More lenient aspect ratio for logos
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0

                        # More accurate logo detection criteria
                        if solidity > 0.7 and w > 20 and h > 20:
                            confidence = min(0.9, solidity * 0.6 + (area / total_area) * 0.4)
                            logos.append({
                                'x': x, 'y': y, 'width': w, 'height': h,
                                'area_ratio': area / total_area,
                                'solidity': solidity,
                                'confidence': confidence,
                                'method': 'contour'
                            })
        except Exception as e:
            print(f"Contour logo detection error: {e}")

        return logos

    def detect_logos_by_features(self, img_array, quality_thresholds):
        """Detect logos using feature detection"""
        logos = []

        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(nfeatures=500)  # Reduced for speed
            kp, des = orb.detectAndCompute(gray, None)

            if des is not None and len(kp) > 5:
                keypoints_xy = np.array([kp.pt for kp in kp])

                if len(keypoints_xy) > 5:
                    clustering = DBSCAN(eps=20, min_samples=3).fit(keypoints_xy)  # Tighter clustering
                    labels = clustering.labels_

                    unique_labels = set(labels)
                    if -1 in unique_labels:
                        unique_labels.remove(-1)

                    for label in unique_labels:
                        cluster_points = keypoints_xy[labels == label]

                        if len(cluster_points) > 3:
                            x_min, y_min = np.min(cluster_points, axis=0)
                            x_max, y_max = np.max(cluster_points, axis=0)

                            width = x_max - x_min
                            height = y_max - y_min

                            max_size_ratio = quality_thresholds.get('logo_max_size_ratio', 0.3)
                            if (15 < width < gray.shape[1] * max_size_ratio and
                                    15 < height < gray.shape[0] * max_size_ratio):
                                density = len(cluster_points) / (width * height)
                                confidence = min(0.8, density * 80)  # Adjusted confidence calculation

                                logos.append({
                                    'x': int(x_min), 'y': int(y_min),
                                    'width': int(width), 'height': int(height),
                                    'keypoint_count': len(cluster_points),
                                    'density': density,
                                    'confidence': confidence,
                                    'method': 'features'
                                })
        except Exception as e:
            print(f"Feature logo detection error: {e}")

        return logos

    def detect_logos_by_template(self, img_array, quality_thresholds):
        """Detect logos using template matching (simplified)"""
        logos = []
        # This would typically match against a database of logo templates
        # For now, return empty list as placeholder
        return logos

    def detect_weapons(self, image, quality_thresholds):
        """Improved weapon detection using shape analysis and text detection"""
        result = {
            'has_weapons': False,
            'weapon_confidence': 0,
            'weapon_details': []
        }

        try:
            if image.mode == 'RGBA':
                image_rgb = Image.new('RGB', image.size, (255, 255, 255))
                image_rgb.paste(image, mask=image.split()[3])
                img_array = np.array(image_rgb)
            else:
                img_array = np.array(image.convert('RGB'))

            if img_array.shape[1] > 600 or img_array.shape[0] > 600:
                scale_factor = min(600 / img_array.shape[1], 600 / img_array.shape[0])
                new_width = int(img_array.shape[1] * scale_factor)
                new_height = int(img_array.shape[0] * scale_factor)
                img_array = cv2.resize(img_array, (new_width, new_height))

            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            weapon_like_contours = []
            threshold = quality_thresholds.get('weapon_confidence_threshold', 0.5)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50:  # Reduced minimum area
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0

                # Improved weapon detection criteria
                weapon_like = False
                confidence = 0.0

                # Gun-like shapes (long and thin)
                if aspect_ratio > 2.5 and solidity > 0.6:
                    weapon_like = True
                    confidence = min(0.7, solidity * 0.5 + (aspect_ratio / 10) * 0.5)

                # Knife-like shapes (triangular)
                elif 0.2 < aspect_ratio < 1.5 and solidity > 0.7:
                    # Check for triangular shape
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if circularity < 0.3:  # Less circular, more angular
                            weapon_like = True
                            confidence = min(0.6, solidity * 0.7)

                if weapon_like and confidence >= threshold:
                    weapon_like_contours.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity,
                        'area': area,
                        'confidence': confidence
                    })

            if weapon_like_contours:
                result['has_weapons'] = True
                result['weapon_confidence'] = max([w['confidence'] for w in weapon_like_contours])
                result['weapon_details'] = weapon_like_contours

            return result
        except Exception as e:
            print(f"Weapon detection error: {e}")
            return result

    def check_pixelation(self, image, quality_thresholds):
        """Check if image is pixelated"""
        try:
            resize_factor = 15  # Balanced for speed and accuracy

            small = image.resize((image.width // resize_factor, image.height // resize_factor), Image.NEAREST)
            enlarged = small.resize((image.width, image.height), Image.NEAREST)

            orig_array = np.array(image.convert('L'))
            enlarged_array = np.array(enlarged.convert('L'))

            if orig_array.shape[0] > 400 or orig_array.shape[1] > 400:
                orig_array = cv2.resize(orig_array, (400, 400))
                enlarged_array = cv2.resize(enlarged_array, (400, 400))

            diff = np.mean(np.abs(orig_array - enlarged_array))
            return bool(diff > quality_thresholds['pixelation_diff_threshold'])
        except:
            return False

    def analyze_colors(self, image):
        """Analyze color distribution in the image"""
        try:
            if image.mode != 'RGB':
                rgb_image = image.convert('RGB')
            else:
                rgb_image = image

            if rgb_image.width > 400 or rgb_image.height > 400:
                rgb_image.thumbnail((400, 400), Image.LANCZOS)

            img_array = np.array(rgb_image)
            mean_color = np.mean(img_array, axis=(0, 1))
            std_color = np.std(img_array, axis=(0, 1))
            is_grayscale = np.allclose(std_color, [0, 0, 0], atol=15)  # More tolerant

            return {
                'mean_rgb': [float(x) for x in mean_color.tolist()],
                'std_rgb': [float(x) for x in std_color.tolist()],
                'is_grayscale': bool(is_grayscale)
            }
        except:
            return {'mean_rgb': [0.0, 0.0, 0.0], 'std_rgb': [0.0, 0.0, 0.0], 'is_grayscale': False}

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
                return 100

            width, height = image.size
            expected_size = width * height * 3
            actual_size = len(image.tobytes())

            if actual_size == 0:
                return 100

            compression_ratio = actual_size / expected_size
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

        # Size check with exact pixel matching
        min_size_pass = (analysis['width'] >= requirements['min_size'] and
                         analysis['height'] >= requirements['min_size'])
        results.append({
            'test': f"Minimum size ({requirements['min_size']}px)",
            'status': 'pass' if min_size_pass else 'fail',
            'message': f"Current: {analysis['width']}x{analysis['height']}px",
            'value': f"{analysis['width']}x{analysis['height']}px",
            'required': f"Min {requirements['min_size']}px"
        })

        max_size_pass = (analysis['width'] <= requirements['max_size'] and
                         analysis['height'] <= requirements['max_size'])
        results.append({
            'test': f"Maximum size ({requirements['max_size']}px)",
            'status': 'pass' if max_size_pass else 'fail',
            'message': f"Current: {analysis['width']}x{analysis['height']}px",
            'value': f"{analysis['width']}x{analysis['height']}px",
            'required': f"Max {requirements['max_size']}px"
        })

        # Exact aspect ratio check
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
            'test': 'Supported format',
            'status': 'pass' if format_pass else 'fail',
            'message': f"Current format: {analysis['format'].upper()}",
            'value': analysis['format'].upper(),
            'required': ', '.join(requirements['formats']).upper()
        })

        # File size check
        file_size_pass = requirements['min_file_size'] <= analysis['size'] <= requirements['max_file_size']
        results.append({
            'test': 'File size',
            'status': 'pass' if file_size_pass else 'fail',
            'message': f"Current size: {self.format_file_size(analysis['size'])}",
            'value': self.format_file_size(analysis['size']),
            'required': f"{self.format_file_size(requirements['min_file_size'])} - {self.format_file_size(requirements['max_file_size'])}"
        })

        # Quality checks
        sharpness_status = 'pass' if analysis['sharpness'] > quality_thresholds['sharpness_min'] else 'fail'
        results.append({
            'test': 'Image sharpness',
            'status': sharpness_status,
            'message': f"Sharpness score: {analysis['sharpness']:.2f}",
            'value': f"{analysis['sharpness']:.2f}",
            'required': f">{quality_thresholds['sharpness_min']}"
        })

        # General requirements with improved logic
        for req_key, req_config in general_requirements.items():
            if not req_config['enabled']:
                continue

            if req_key == 'no_watermarks':
                status = 'pass' if not analysis['has_watermark'] else 'fail'
                message = 'No watermarks found' if not analysis['has_watermark'] else 'Watermarks detected'
            elif req_key == 'no_copyright_text' or req_key == 'no_website_urls' or req_key == 'no_social_handles':
                has_forbidden_text = analysis.get('forbidden_text_found', False)
                is_artistic_text = analysis.get('text_analysis', {}).get('is_artistic', False)
                status = 'pass' if not has_forbidden_text or is_artistic_text else 'fail'
                message = 'No forbidden text' if not has_forbidden_text or is_artistic_text else 'Forbidden text detected'
            elif req_key == 'no_blurry_images':
                status = 'pass' if not analysis['is_blurry'] else 'fail'
                message = 'Image is sharp' if not analysis['is_blurry'] else 'Image is blurry'
            elif req_key == 'no_pixelated_images':
                status = 'pass' if not analysis['is_pixelated'] else 'fail'
                message = 'Image is not pixelated' if not analysis['is_pixelated'] else 'Image is pixelated'
            elif req_key == 'no_logos':
                has_logos = analysis['has_logos']
                is_artistic_text = analysis.get('text_analysis', {}).get('is_artistic', False)
                status = 'pass' if not has_logos or is_artistic_text else 'fail'
                message = 'No logos found' if not has_logos or is_artistic_text else 'Logos detected'
            elif req_key == 'no_bad_words':
                has_bad_words = analysis.get('bad_words_found', False)
                is_artistic_text = analysis.get('text_analysis', {}).get('is_artistic', False)
                status = 'pass' if not has_bad_words or is_artistic_text else 'fail'
                message = 'No offensive language' if not has_bad_words or is_artistic_text else 'Offensive language detected'
            elif req_key == 'no_weapons':
                has_weapons = analysis.get('has_weapons', False)
                has_weapon_words = analysis.get('weapon_words_found', False)
                status = 'pass' if not has_weapons and not has_weapon_words else 'fail'
                message = 'No weapons content' if not has_weapons and not has_weapon_words else 'Weapons content detected'
            elif req_key == 'has_artist_info':
                has_artist_info = analysis.get('artist_info_found', False)
                status = 'pass' if has_artist_info else 'warning'  # Warning instead of fail for artist info
                message = 'Artist info found' if has_artist_info else 'Artist info not detected'
            else:
                continue

            results.append({
                'test': req_config['description'],
                'status': status,
                'message': message,
                'value': 'Yes' if status == 'pass' else 'No',
                'required': req_config['description']
            })

        return results

    def format_file_size(self, bytes):
        if bytes == 0:
            return "0 Bytes"
        units = ["Bytes", "KB", "MB", "GB"]
        i = int(np.floor(np.log(bytes) / np.log(1024)))
        return f"{bytes / (1024 ** i):.2f} {units[i]}"

    def determine_approval_status(self, platform_results, platform):
        """Determine if artwork is approved (80%+ score) or rejected"""
        rules = self.get_rules()
        platform_req = rules['platform_requirements'].get(platform, {})
        approval_threshold = platform_req.get('approval_threshold', 80)

        all_statuses = [result['status'] for result in platform_results]
        total_tests = len(all_statuses)

        if total_tests == 0:
            return {'status': 'fail', 'score': 0, 'approved': False}

        pass_count = all_statuses.count('pass')
        warning_count = all_statuses.count('warning')

        # Calculate score (pass = 1, warning = 0.5, fail = 0)
        score_percentage = (pass_count + (warning_count * 0.5)) / total_tests * 100
        score_percentage = round(score_percentage, 1)

        # Check for critical failures (any fail status means rejection)
        has_critical_failures = 'fail' in all_statuses

        # Approved if score >= threshold AND no critical failures
        approved = score_percentage >= approval_threshold and not has_critical_failures

        return {
            'status': 'pass' if approved else 'fail',
            'score': score_percentage,
            'approved': approved,
            'pass_count': pass_count,
            'warning_count': warning_count,
            'fail_count': all_statuses.count('fail'),
            'total_tests': total_tests,
            'threshold': approval_threshold
        }

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

        # Aspect ratio recommendations
        if abs(analysis['aspect_ratio'] - 1.0) > 0.01:
            recommendations.append({
                'priority': 'high',
                'message': 'Crop image to perfect square (1:1 aspect ratio)',
                'category': 'composition'
            })

        # Content recommendations
        if analysis['has_watermark']:
            recommendations.append({
                'priority': 'high',
                'message': 'Remove any watermarks from the artwork',
                'category': 'content'
            })

        if (analysis['has_text'] and analysis.get('forbidden_text_found', False) and
                not analysis.get('text_analysis', {}).get('is_artistic', False)):
            recommendations.append({
                'priority': 'high',
                'message': 'Remove any text, URLs, or social media handles from the artwork',
                'category': 'content'
            })

        if (analysis['has_logos'] and
                not analysis.get('text_analysis', {}).get('is_artistic', False)):
            recommendations.append({
                'priority': 'high',
                'message': 'Remove any logos or branding elements from the artwork',
                'category': 'content'
            })

        if (analysis.get('bad_words_found', False) and
                not analysis.get('text_analysis', {}).get('is_artistic', False)):
            recommendations.append({
                'priority': 'high',
                'message': 'Remove offensive or inappropriate language from the artwork',
                'category': 'content'
            })

        if analysis.get('has_weapons', False) or analysis.get('weapon_words_found', False):
            recommendations.append({
                'priority': 'high',
                'message': 'Remove weapons or violent content from the artwork',
                'category': 'content'
            })

        if not analysis.get('artist_info_found', False):
            recommendations.append({
                'priority': 'medium',
                'message': 'Consider adding artist/album/song information',
                'category': 'content'
            })

        return recommendations

    def generate_summary_report(self, analysis, platform_results, recommendations, approval_statuses):
        """Generate a comprehensive summary report"""
        summary = {
            'overall_eligibility': self.determine_overall_eligibility(analysis, platform_results, approval_statuses),
            'key_findings': [],
            'content_analysis': {},
            'technical_analysis': {},
            'platform_compatibility': {}
        }

        # Key findings
        if analysis.get('has_weapons', False) or analysis.get('weapon_words_found', False):
            summary['key_findings'].append({
                'type': 'critical',
                'message': 'Weapons/violent content detected',
                'confidence': analysis.get('weapon_confidence', 0)
            })

        if analysis.get('bad_words_found', False):
            summary['key_findings'].append({
                'type': 'critical',
                'message': 'Offensive language detected'
            })

        if analysis.get('has_watermark', False):
            summary['key_findings'].append({
                'type': 'critical',
                'message': 'Watermarks detected'
            })

        if analysis.get('forbidden_text_found', False):
            summary['key_findings'].append({
                'type': 'critical',
                'message': 'Forbidden text detected'
            })

        if analysis.get('artist_info_found', False):
            summary['key_findings'].append({
                'type': 'positive',
                'message': 'Artist/album information found'
            })

        # Content analysis
        summary['content_analysis'] = {
            'text_detected': analysis.get('has_text', False),
            'text_count': len(analysis.get('detected_text', [])),
            'bad_words_found': analysis.get('bad_words_found', False),
            'weapons_detected': analysis.get('has_weapons', False) or analysis.get('weapon_words_found', False),
            'weapon_confidence': analysis.get('weapon_confidence', 0),
            'watermarks_detected': analysis.get('has_watermark', False),
            'logos_detected': analysis.get('has_logos', False),
            'logo_count': analysis.get('logo_count', 0),
            'artist_info_found': analysis.get('artist_info_found', False)
        }

        # Technical analysis
        summary['technical_analysis'] = {
            'dimensions': f"{analysis['width']}x{analysis['height']}",
            'aspect_ratio': round(analysis['aspect_ratio'], 3),
            'file_size': self.format_file_size(analysis['size']),
            'format': analysis['format'].upper(),
            'color_mode': analysis['mode'],
            'dpi': analysis['dpi'],
            'sharpness': round(analysis['sharpness'], 2),
            'brightness': round(analysis['brightness'], 2),
            'contrast': round(analysis['contrast'], 2),
            'compression_quality': analysis.get('compression_quality', 0),
            'has_transparency': analysis.get('has_transparency', False),
            'is_blurry': analysis.get('is_blurry', False),
            'is_pixelated': analysis.get('is_pixelated', False)
        }

        # Platform compatibility with approval status
        for platform, status in approval_statuses.items():
            summary['platform_compatibility'][platform] = {
                'approved': status['approved'],
                'status': status['status'],
                'score': status['score'],
                'threshold': status['threshold'],
                'pass_count': status['pass_count'],
                'fail_count': status['fail_count'],
                'warning_count': status['warning_count']
            }

        return summary

    def determine_overall_eligibility(self, analysis, platform_results, approval_statuses):
        """Determine if artwork is eligible for release"""
        critical_issues = []

        if analysis.get('has_weapons', False) or analysis.get('weapon_words_found', False):
            critical_issues.append('Weapons/violent content detected')

        if analysis.get('bad_words_found', False) and not analysis.get('text_analysis', {}).get('is_artistic', False):
            critical_issues.append('Offensive language detected')

        if analysis.get('has_watermark', False):
            critical_issues.append('Watermarks detected')

        # Check platform approvals
        platform_approvals = {}
        for platform, status in approval_statuses.items():
            platform_approvals[platform] = status['approved']

        if critical_issues:
            return {
                'eligible': False,
                'reason': 'Critical content issues prevent approval',
                'issues': critical_issues,
                'platform_approvals': platform_approvals
            }
        else:
            # Check if any platform approves the artwork
            any_approved = any(status['approved'] for status in approval_statuses.values())
            if any_approved:
                return {
                    'eligible': True,
                    'reason': 'Artwork meets requirements for at least one platform',
                    'issues': [],
                    'platform_approvals': platform_approvals
                }
            else:
                return {
                    'eligible': False,
                    'reason': 'Does not meet platform requirements',
                    'issues': [],
                    'platform_approvals': platform_approvals
                }


analyzer = AdvancedArtworkAnalyzer()


@app.route('/')
def index():
    return render_template('index.html')


def make_serializable(obj):
    """Convert non-serializable objects to serializable formats"""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
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
        approval_statuses = {}

        rules = analyzer.get_rules()
        for platform, config in rules['platform_requirements'].items():
            platform_results[platform] = analyzer.check_requirements(analysis, platform)

            if config['enabled']:
                recommendations[platform] = analyzer.generate_recommendations(analysis, platform_results[platform])
                approval_statuses[platform] = analyzer.determine_approval_status(platform_results[platform], platform)
            else:
                recommendations[platform] = []
                approval_statuses[platform] = {'status': 'disabled', 'approved': False, 'score': 0}

        # Generate summary report
        summary_report = analyzer.generate_summary_report(analysis, platform_results, recommendations,
                                                          approval_statuses)

        # Prepare response
        response = {
            'analysis': make_serializable(analysis),
            'platform_results': make_serializable(platform_results),
            'recommendations': make_serializable(recommendations),
            'approval_statuses': approval_statuses,
            'summary_report': summary_report,
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
                'processing_time': f"{analysis.get('processing_time', 0):.2f}s"
            }
        }

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify(response)


# Admin routes
@app.route('/admin')
def admin_dashboard():
    rules = load_rules()
    return render_template('admin_dashboard.html', rules=rules)


@app.route('/admin/rules/platform/<platform>', methods=['GET', 'POST'])
def edit_platform_rules(platform):
    rules = load_rules()
    if platform not in rules['platform_requirements']:
        return "Platform not found", 404

    if request.method == 'POST':
        platform_rules = rules['platform_requirements'][platform]
        platform_rules['min_size'] = int(request.form.get('min_size', platform_rules['min_size']))
        platform_rules['max_size'] = int(request.form.get('max_size', platform_rules['max_size']))
        platform_rules['aspect_ratio'] = float(request.form.get('aspect_ratio', platform_rules['aspect_ratio']))
        platform_rules['max_file_size'] = int(request.form.get('max_file_size', platform_rules['max_file_size']))
        platform_rules['min_file_size'] = int(request.form.get('min_file_size', platform_rules['min_file_size']))
        platform_rules['dpi'] = int(request.form.get('dpi', platform_rules['dpi']))
        platform_rules['approval_threshold'] = int(
            request.form.get('approval_threshold', platform_rules.get('approval_threshold', 80)))

        formats = request.form.get('formats', '').split(',')
        platform_rules['formats'] = [f.strip().lower() for f in formats if f.strip()]
        platform_rules['color_space'] = request.form.get('color_space', platform_rules['color_space'])
        platform_rules['name'] = request.form.get('name', platform_rules['name'])
        platform_rules['enabled'] = request.form.get('enabled') == 'on'

        save_rules(rules)
        analyzer.reload_rules()
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_platform_rules.html',
                           platform=platform,
                           rules=rules['platform_requirements'][platform])


@app.route('/admin/rules/general', methods=['GET', 'POST'])
def edit_general_rules():
    rules = load_rules()

    if request.method == 'POST':
        for req_key in rules['general_requirements']:
            rules['general_requirements'][req_key]['enabled'] = request.form.get(f"{req_key}_enabled") == 'on'

        # Update quality thresholds
        for threshold_key in rules['quality_thresholds']:
            if threshold_key in request.form:
                try:
                    if '.' in request.form[threshold_key]:
                        rules['quality_thresholds'][threshold_key] = float(request.form[threshold_key])
                    else:
                        rules['quality_thresholds'][threshold_key] = int(request.form[threshold_key])
                except ValueError:
                    pass

        save_rules(rules)
        analyzer.reload_rules()
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_general_rules.html', rules=rules)


@app.route('/admin/rules/text', methods=['GET', 'POST'])
def edit_text_rules():
    rules = load_rules()

    if request.method == 'POST':
        # Update text detection settings
        text_rules = rules['text_detection']
        text_rules['min_text_length'] = int(request.form.get('min_text_length', text_rules['min_text_length']))
        text_rules['max_text_density'] = float(request.form.get('max_text_density', text_rules['max_text_density']))

        # Update forbidden terms
        forbidden_terms = request.form.get('forbidden_terms', '').split('\n')
        text_rules['forbidden_terms'] = [term.strip() for term in forbidden_terms if term.strip()]

        # Update performance settings
        perf_settings = rules['performance_settings']
        perf_settings['max_image_size'] = int(request.form.get('max_image_size', perf_settings['max_image_size']))
        perf_settings['logo_detection_enabled'] = request.form.get('logo_detection_enabled') == 'on'
        perf_settings['text_detection_enabled'] = request.form.get('text_detection_enabled') == 'on'
        perf_settings['watermark_detection_enabled'] = request.form.get('watermark_detection_enabled') == 'on'
        perf_settings['weapon_detection_enabled'] = request.form.get('weapon_detection_enabled') == 'on'

        save_rules(rules)
        analyzer.reload_rules()
        return redirect(url_for('admin_dashboard'))

    return render_template('edit_text_rules.html', rules=rules)


@app.route('/admin/rules/reset', methods=['POST'])
def reset_rules():
    save_rules(DEFAULT_RULES)
    analyzer.reload_rules()
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/add_platform', methods=['POST'])
def add_platform():
    rules = load_rules()
    platform_name = request.form.get('platform_name')
    platform_key = platform_name.lower().replace(' ', '_')

    if platform_key not in rules['platform_requirements']:
        rules['platform_requirements'][platform_key] = {
            "min_size": 1400,
            "max_size": 3000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "dpi": 72,
            "color_space": "RGB",
            "name": platform_name,
            "enabled": True,
            "approval_threshold": 80
        }
        save_rules(rules)
        analyzer.reload_rules()

    return redirect(url_for('admin_dashboard'))


@app.route('/admin/remove_platform/<platform>', methods=['POST'])
def remove_platform(platform):
    rules = load_rules()
    if platform in rules['platform_requirements'] and platform not in ['spotify', 'apple']:
        del rules['platform_requirements'][platform]
        save_rules(rules)
        analyzer.reload_rules()
    return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)

    # Create static directory if it doesn't exist
    os.makedirs('static', exist_ok=True)

    load_rules()
    app.run(debug=True, port=5000)