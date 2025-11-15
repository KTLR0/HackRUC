import os
from dotenv import load_dotenv
import cv2
from flask import Flask, request, jsonify, render_template # <-- ADD render_template
import yt_dlp
import google.generativeai as genai
import moviepy.editor as mp

# --- CONFIGURATION ---
load_dotenv() # <-- Ensure .env variables are loaded

try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    raise Exception("GEMINI_API_KEY environment variable not set. Please create a .env file and set it.")

HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml'
if not os.path.exists(HAAR_CASCADE_PATH):
    raise FileNotFoundError(f"{HAAR_CASCADE_PATH} not found. Please make sure it's in the same directory as app.py.")
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)


# --- FLASK APP INITIALIZATION ---
# The template_folder tells Flask where to look for your HTML files.
app = Flask(__name__, template_folder='static')


# --- HELPER FUNCTIONS (Your original functions are perfect, no changes needed here) ---

def download_video(youtube_url):
    print(f"Downloading video from: {youtube_url}")
    video_filename = 'temp_video.mp4'
    ydl_opts = {
        'format': 'best[ext=mp4][height<=720]/best[height<=720]', # Limit to 720p for faster processing
        'outtmpl': video_filename,
        'overwrites': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    video_path = os.path.abspath(video_filename)
    print(f"Video downloaded to: {video_path}")
    return video_path

def analyze_video_for_faces(video_path):
    print("Analyzing video for faces...")
    vidcap = cv2.VideoCapture(video_path)
    total_frames = 0
    frames_with_faces = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        total_frames += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            frames_with_faces += 1
    vidcap.release()
    if total_frames == 0:
        return 0
    enthusiasm_score = (frames_with_faces / total_frames) * 100
    print(f"Analysis complete. Score: {enthusiasm_score:.2f}")
    return enthusiasm_score

def get_gemini_evaluation(score):
    print("Getting evaluation from Gemini...")
    model = genai.GenerativeModel('gemini-2.0-flash') # Using 'gemini-pro' as it's a solid choice.
    prompt = f"""
    Based on a facial analysis of a person in a YouTube video, a metric called 'face presence' was calculated.
    This metric represents the percentage of video frames in which a face was clearly visible and detected.
    The calculated 'face presence' score is {score:.2f}%.

    Assuming that higher 'face presence' correlates with higher engagement and enthusiasm (e.g., the person is consistently facing the camera and is the focus), please provide a brief, one-paragraph evaluation of their likely enthusiasm level.
    """
    try:
        response = model.generate_content(prompt)
        print("Gemini evaluation received.")
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Could not generate an AI evaluation due to an API error."
    
# video splitting
def extract_video_no_audio(input_path, output_path="video_no_audio.mp4"):
    clip = mp.VideoFileClip(input_path)
    clip_no_audio = clip.without_audio()
    clip_no_audio.write_videofile(output_path, codec="libx264")
    clip.close()
    clip_no_audio.close()
    return os.path.abspath(output_path)

def extract_audio(input_path, output_path="audio_only.mp3"):
    clip = mp.VideoFileClip(input_path)
    clip.audio.write_audiofile(output_path)
    clip.close()
    return os.path.abspath(output_path)


# --- FLASK API ROUTES ---

# NEW: This route will serve your main HTML page
@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')

# CHANGED: The route now matches the JavaScript fetch() call
@app.route('/analyze_video/', methods=['POST'])
def analyze_video_route():
    data = request.get_json()
    if not data or 'video_url' not in data:
        return jsonify({'error': 'video_url is required'}), 400

    video_url = data['video_url']
    video_path = None
    
    try:
        # 1. Download original video
        video_path = download_video(video_url)

        # 2. Create video-only file
        video_no_audio_path = extract_video_no_audio(video_path)

        # 3. Create audio-only file
        audio_path = extract_audio(video_path)

        # 4. Get transcript
        # transcript = extract_transcript(video_url)

        # 5. Your existing analysis
        score = analyze_video_for_faces(video_path)
        evaluation = get_gemini_evaluation(score)

        return jsonify({
            "enthusiasm_score": score,
            "gemini_evaluation": evaluation,
            "video_no_audio": video_no_audio_path,
            "audio_file": audio_path,
            # "transcript": transcript
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup original video (we keep the split files)
        if video_path and os.path.exists(video_path):
            os.remove(video_path)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)