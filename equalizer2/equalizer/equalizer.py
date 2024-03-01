import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import librosa
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav, mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Check if the uploaded file has a valid extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Apply equalizer settings to the audio
def apply_equalizer(audio_path, bass, midrange, treble):
    y, sr = librosa.load(audio_path, sr=None)
    D = np.abs(librosa.stft(y))
    equalizer = np.array([1 + (bass - 50) / 100, 1 + (midrange - 50) / 100, 1 + (treble - 50) / 100])
    D *= equalizer[:, np.newaxis]
    y_eq = librosa.istft(D)
    return y_eq

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('equalize', filename=filename))
    else:
        return redirect(request.url)

@app.route('/equalize/<filename>')
def equalize(filename):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    bass = int(request.args.get('bass', 50))
    midrange = int(request.args.get('midrange', 50))
    treble = int(request.args.get('treble', 50))

    # Apply equalizer settings
    y_eq = apply_equalizer(audio_path, bass, midrange, treble)

    # Save the equalized audio
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'equalized_' + filename)
    librosa.output.write_wav(output_path, y_eq, )

    return render_template('equalized.html', filename='equalized_' + filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
