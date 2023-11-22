def Transcribe(audio_file):
    import whisper
    import warnings
    import os
    warnings.simplefilter("ignore")

    model = whisper.load_model("tiny.en")
    
    
    result = model.transcribe(audio=f"{audio_file}")
    
    
    return result["text"]

