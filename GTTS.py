from gtts import gTTS
import pygame
def gtts(text,outputpath=r"C:\Users\aaron\OneDrive\Desktop\Audio.mp3"):
    language='en'
    tts=gTTS(text=text,lang=language,slow=False)
    tts.save(outputpath)
    pygame.mixer.init()
    pygame.mixer.music.load(outputpath)
    while pygame.mixer.music.get_busy():
        pass
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()

    