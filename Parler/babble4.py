import torch
import torchaudio
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from ai_service_factory import AIServiceFactory
from config_environment_module import ConfigEnvironmentModule
import simpleaudio as sa
import numpy as np
import re
import nltk.data
import librosa
from sklearn.cluster import KMeans

# Download Punkt sentence tokenizer data if not already downloaded
try:
    nltk.data.find("tokenizers/punkt/english.pickle")
except LookupError:
    nltk.download("punkt")

tokenizer_cache = {}


def generate_text(prompt, service):
    response = service.send_request(prompt)
    if isinstance(response, str):
        return response.strip()
    elif hasattr(response, "choices") and response.choices:
        return response.choices[0].text.strip()
    elif hasattr(response, "text"):
        return response.text.strip()
    else:
        return ""


def text_to_speech(text, speaker_description, model, tokenizer, device):
    with torch.no_grad():
        if speaker_description not in tokenizer_cache:
            tokenizer_cache[speaker_description] = tokenizer(
                speaker_description, return_tensors="pt"
            ).input_ids.to(device)
        input_ids = tokenizer_cache[speaker_description]
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        with torch.cuda.device(device):
            generation = model.generate(
                input_ids=input_ids, prompt_input_ids=prompt_input_ids
            )
            audio_arr = generation.cpu().numpy().squeeze().astype(np.float32)
        del generation
        return audio_arr, model.config.sampling_rate


def split_audio_by_silence(audio_data, sampling_rate, min_silence_len=2000):
    """Splits audio with adaptive thresholding and spectral analysis."""

    # Noise reduction and normalization
    audio_data = librosa.effects.trim(audio_data)[0]
    audio_data /= np.max(np.abs(audio_data))

    # Spectral centroid as a feature, using keyword arguments
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate)[0]

    # Adaptive thresholding and silence detection
    frames = librosa.util.frame(audio_data, frame_length=1024, hop_length=256)
    frames_energy = np.sum(frames**2, axis=0)
    silence_threshold = np.mean(frames_energy) / 4  # Adjust factor as needed
    is_silence = frames_energy < silence_threshold

    # Ensure frames_energy and spectral_centroids have the same length
    min_length = min(len(frames_energy), len(spectral_centroids))
    frames_energy = frames_energy[:min_length]
    spectral_centroids = spectral_centroids[:min_length]

    # Combine energy and spectral information
    combined_features = np.vstack((frames_energy, spectral_centroids))
    kmeans = KMeans(n_clusters=2)  # Assuming 2 clusters: speech and silence
    kmeans.fit(combined_features.T)
    labels = kmeans.labels_

    # Segment boundaries based on cluster labels and minimum silence length
    segments = []
    start = 0
    for i, label in enumerate(labels):
        if i > 0 and labels[i] != labels[i - 1] and i - start > min_silence_len:
            segments.append(audio_data[start * 512 : i * 512])
            start = i

    if len(audio_data) > start * 512:
        segments.append(audio_data[start * 512 :])

    return segments


def generate_conversation(topic, num_iterations, openai_service):
    prompt = f"Let's talk about {topic}."
    full_conversation = ""
    for _ in range(num_iterations):
        alice_text = generate_text(prompt, openai_service)
        full_conversation += f"Alice: {alice_text}\n"
        bob_text = generate_text(alice_text, openai_service)
        full_conversation += f"Bob: {bob_text}\n"
        prompt = bob_text
    return full_conversation


def split_conversation(full_conversation):
    sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
    alice_sentences = sent_detector.tokenize(
        " ".join(re.findall(r"Alice: (.*?)\n", full_conversation))
    )
    bob_sentences = sent_detector.tokenize(
        " ".join(re.findall(r"Bob: (.*?)\n", full_conversation))
    )
    alice_text = " ".join(alice_sentences)
    bob_text = " ".join(bob_sentences)
    return alice_text, bob_text


def interleave_audio_segments(alice_segments, bob_segments, sampling_rate, pause_duration=0.5):
    silence = np.zeros((int(sampling_rate * pause_duration),), dtype=np.float32)
    combined_audio = []

    max_segments = max(len(alice_segments), len(bob_segments))
    for i in range(max_segments):
        if i < len(alice_segments):
            combined_audio.append(alice_segments[i])
            combined_audio.append(silence)
        if i < len(bob_segments):
            combined_audio.append(bob_segments[i])
            combined_audio.append(silence)

    return np.concatenate(combined_audio, axis=0)

def play_audio(audio_data, sampling_rate):
    audio_data = (audio_data * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio_data, 1, 2, sampling_rate)
    play_obj.wait_done()


def main():
    config = ConfigEnvironmentModule()
    try:
        openai_service = AIServiceFactory.get_ai_service()
    except (ValueError, Exception) as e:
        print(f"Error getting AI service: {e}")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "parler-tts/parler_tts_mini_v0.1", torch_dtype=torch.float32
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler_tts_mini_v0.1")
    topic = input("Enter the conversation topic: ")
    alice_description = "A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
    bob_description = "A male speaker with a calm and knowledgeable voice."
    num_iterations = int(input("Enter the number of conversation iterations: "))

    # Generate the entire conversation using OpenAI
    full_conversation = generate_conversation(topic, num_iterations, openai_service)

    # Split the conversation into separate text for Alice and Bob
    alice_text, bob_text = split_conversation(full_conversation)

    # Generate audio for each character in a single TTS call
    alice_audio, sampling_rate = text_to_speech(
        alice_text, alice_description, model, tokenizer, device
    )
    bob_audio, _ = text_to_speech(
        bob_text, bob_description, model, tokenizer, device
    )

    # Analyze the audio files to determine silence segments
    alice_segments = split_audio_by_silence(alice_audio, sampling_rate)
    bob_segments = split_audio_by_silence(bob_audio, sampling_rate)

    # Interleave the audio segments with appropriate pauses
    combined_audio = interleave_audio_segments(alice_segments, bob_segments, sampling_rate)

    # Save and play the conversation
    sf.write("conversation.wav", combined_audio, sampling_rate, subtype="PCM_16")
    print("Full conversation saved to conversation.wav")

    # Play the audio
    play_audio(combined_audio, sampling_rate)


if __name__ == "__main__":
    main()