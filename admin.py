from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Path to the rules configuration file
RULES_FILE = "artwork_rules.json"

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
        },
        "youtube": {
            "min_size": 2560,
            "max_size": 4096,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": "YouTube Music",
            "enabled": True
        },
        "amazon": {
            "min_size": 1400,
            "max_size": 3000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": "Amazon Music",
            "enabled": True
        },
        "deezer": {
            "min_size": 1400,
            "max_size": 4000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": "Deezer",
            "enabled": True
        },
        "tidal": {
            "min_size": 1600,
            "max_size": 3000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": "Tidal",
            "enabled": True
        },
        "soundcloud": {
            "min_size": 1000,
            "max_size": 5000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": "SoundCloud",
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
        },
        "no_borders_frames": {
            "description": "No borders or frames",
            "severity": "medium",
            "enabled": True
        },
        "no_collages": {
            "description": "Not a collage of multiple images",
            "severity": "medium",
            "enabled": True
        },
        "no_placeholder_images": {
            "description": "No placeholder or stock images",
            "severity": "high",
            "enabled": True
        },
        "no_screenshot_images": {
            "description": "No screenshots",
            "severity": "high",
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
        "pixelation_diff_threshold": 30,
        "color_variance_threshold": 10,
        "saturation_min": 20,
        "saturation_max": 200
    },
    "text_detection": {
        "forbidden_terms": [
            "copyright", "©", "®", "™", "www.", ".com", ".net", ".org",
            "@", "instagram", "facebook", "twitter", "tiktok", "youtube",
            "soundcloud", "spotify", "apple music", "amazon music", "deezer",
            "tidal", "promo", "free download", "explicit", "sample", "unauthorized"
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
        thresholds['color_variance_threshold'] = float(
            request.form.get('color_variance_threshold', thresholds['color_variance_threshold']))
        thresholds['saturation_min'] = float(request.form.get('saturation_min', thresholds['saturation_min']))
        thresholds['saturation_max'] = float(request.form.get('saturation_max', thresholds['saturation_max']))

        # Save updated rules
        save_rules(rules)

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

        return redirect(url_for('admin_dashboard'))

    return render_template('edit_text_rules.html', text_rules=rules['text_detection'])


@app.route('/admin/rules/add_platform', methods=['POST'])
def add_platform():
    """Add a new platform"""
    rules = load_rules()

    platform_key = request.form.get('platform_key', '').strip().lower()
    platform_name = request.form.get('platform_name', '').strip()

    if platform_key and platform_name and platform_key not in rules['platform_requirements']:
        # Create new platform with default values
        rules['platform_requirements'][platform_key] = {
            "min_size": 1400,
            "max_size": 3000,
            "aspect_ratio": 1.0,
            "formats": ["jpeg", "jpg", "png"],
            "max_file_size": 10 * 1024 * 1024,
            "min_file_size": 100 * 1024,
            "color_space": "RGB",
            "dpi": 72,
            "name": platform_name,
            "enabled": True
        }

        # Save updated rules
        save_rules(rules)

    return redirect(url_for('admin_dashboard'))


@app.route('/admin/rules/remove_platform/<platform>')
def remove_platform(platform):
    """Remove a platform"""
    rules = load_rules()

    if platform in rules['platform_requirements']:
        del rules['platform_requirements'][platform]
        save_rules(rules)

    return redirect(url_for('admin_dashboard'))


@app.route('/admin/rules/reset', methods=['POST'])
def reset_rules():
    """Reset all rules to defaults"""
    save_rules(DEFAULT_RULES)
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/rules/export')
def export_rules():
    """Export rules as JSON file"""
    rules = load_rules()
    return jsonify(rules)


@app.route('/admin/rules/import', methods=['POST'])
def import_rules():
    """Import rules from JSON file"""
    if 'rules_file' not in request.files:
        return "No file uploaded", 400

    file = request.files['rules_file']
    if file.filename == '':
        return "No file selected", 400

    if file and file.filename.endswith('.json'):
        try:
            rules = json.load(file)
            save_rules(rules)
            return redirect(url_for('admin_dashboard'))
        except:
            return "Invalid JSON file", 400

    return "Invalid file format", 400


if __name__ == '__main__':
    # Ensure rules file exists
    load_rules()

    # Run the admin panel on a different port
    app.run(debug=True, port=5001)