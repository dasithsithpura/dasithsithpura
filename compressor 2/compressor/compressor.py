from flask import Flask, request, render_template, send_file, jsonify
import librosa
import numpy as np
import soundfile as sf
import os

app = Flask(__name__)

# Define default parameters
DEFAULT_THRESHOLD = -20  # dBFS
DEFAULT_RATIO = 4
DEFAULT_ATTACK_TIME = 0.1  # seconds
DEFAULT_RELEASE_TIME = 0.5  # seconds
DEFAULT_GAIN = 0  # dB

def read_audio(input_file):
    try:
        # Load audio file
        y, sr = librosa.load(input_file, sr=None)
        return y, sr
    except Exception as e:
        return None, None

def divide_into_frames(y, frame_size):
    # Divide audio into frames
    frames = []
    num_frames = len(y) // frame_size
    for i in range(num_frames):
        frame = y[i * frame_size : (i + 1) * frame_size]
        frames.append(frame)
    return frames

def calculate_gain_reduction(frame, threshold, ratio):
    # Calculate gain reduction for a single frame
    rms = np.sqrt(np.mean(frame**2))
    if rms > threshold:
        gain_reduction = (rms - threshold) / ratio
    else:
        gain_reduction = 0
    return gain_reduction

def apply_attack_and_release(gain_reduction, attack_time, release_time, previous_gain_reduction=0):
    # Apply attack and release
    if gain_reduction > previous_gain_reduction:
        gain_reduction = previous_gain_reduction + (attack_time * (gain_reduction - previous_gain_reduction))
    else:
        gain_reduction = previous_gain_reduction + (release_time * (gain_reduction - previous_gain_reduction))
    return gain_reduction

def apply_makeup_gain(frame, gain_reduction, gain):
    # Apply makeup gain
    makeup_gain = np.power(10, (gain_reduction + gain) / 20)  # Convert gain reduction to linear scale
    return frame * makeup_gain

def output_audio(compressed_frames, output_file, sr):
    # Concatenate frames and save the compressed audio
    compressed_audio = np.concatenate(compressed_frames)
    sf.write(output_file, compressed_audio, sr)

@app.route('/')
def index():
    return render_template('compressor.html')

@app.route('/compress', methods=['POST'])
def compress():
    try:
        # Get uploaded file
        audio_file = request.files['audio_file']

        # Get parameters from the form or use defaults
        threshold_db = float(request.form.get('threshold', DEFAULT_THRESHOLD))
        ratio = float(request.form.get('ratio', DEFAULT_RATIO))
        attack_time = float(request.form.get('attack_time', DEFAULT_ATTACK_TIME))
        release_time = float(request.form.get('release_time', DEFAULT_RELEASE_TIME))
        gain = float(request.form.get('gain', DEFAULT_GAIN))

        # Save the uploaded file
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create the upload folder if it doesn't exist

        input_file = os.path.join(upload_folder, audio_file.filename)
        audio_file.save(input_file)

        # Parameters
        frame_size = 1024

        # Read Audio Input
        y, sr = read_audio(input_file)

        if y is None or sr is None:
            return jsonify({'error': 'Failed to read audio file.'}), 500

        # Divide Audio into Frames
        frames = divide_into_frames(y, frame_size)

        # Calculate Gain Reduction
        compressed_frames = []
        previous_gain_reduction = 0
        for frame in frames:
            gain_reduction = calculate_gain_reduction(frame, threshold_db, ratio)

            # Apply Attack and Release
            gain_reduction = apply_attack_and_release(gain_reduction, attack_time, release_time, previous_gain_reduction)
            previous_gain_reduction = gain_reduction

            # Apply Makeup Gain
            compressed_frame = apply_makeup_gain(frame, -gain_reduction, gain)  # Include the gain parameter here
            compressed_frames.append(compressed_frame)

        # Output the Compressed Audio
        output_folder = 'output'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_file = os.path.abspath(os.path.join(output_folder, 'compressed_audio.wav'))
        output_audio(compressed_frames, output_file, sr)

        return send_file(output_file, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
