import cv2
import time
from GAN.generate import Generator
from peekingduck_process.preprocess_image import Preprocessor
import torch
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import pyttsx3
engine = pyttsx3.init()

if torch.cuda.is_available():  
    device = "cuda:0"
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:  
    device = "cpu" 

disclaimer = "This app is powered by an artifical intelligence that generates text, be it compliments \nor insults. The developers have put it utmost effort to ensure insults do not cause excessive \nharm, but we are unable to control what text may be displayed. By continuing to use this app, \nyou acknowledge the possibility that potentially hurtful text may be generated."


G = Generator(10).to(device)
# load pre-trained model
preprocess=Preprocessor()
vid = cv2.VideoCapture(0)
cv2.namedWindow("frame")

def generateTextFromImage(image, attitudes):
    image, features=preprocess(image)
    if features:
        features=torch.tensor(features[0],dtype=torch.float).unsqueeze(0)
    else:
        return None
    toks = G.forward(features, attitudes, max_length=20,temperature=0.1,return_probs=True)[0]
    final_text=G.tokens.batch_decode(toks,skip_special_tokens=True)
    return "".join(final_text)

frame = []
keep_running = True
attitude = 0
skip_disclaimer = False
while keep_running:
    if not skip_disclaimer:
        ret, frame = vid.read()
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("Arial.ttf", 20)
        bbox = draw.textbbox((10, 10), disclaimer, font=font)
        draw.rectangle(bbox, fill="white")
        draw.text((10, 10), disclaimer, font=font, fill="#000000")
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', cv2_im_processed)
        skip_disclaimer = True
        k = cv2.waitKey(10)
        time.sleep(5)
    while keep_running:
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            keep_running = False
            break
        if k == ord('c'):
            attitude = 1.0
            break
        if k == ord('i'):
            attitude = -1.0
            break
    if keep_running:
        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("Arial.ttf", 20)

        display_text = "Generating compliment..." if (attitude>0) else "Generating insult..."
        draw.text((10, 10), display_text, font=font, fill="#000000")
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', cv2_im_processed)
        k = cv2.waitKey(1)

        cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        draw = ImageDraw.Draw(pil_im)
        txt = generateTextFromImage(pil_im, torch.tensor([[attitude]]))
        print("\n", txt, "\n", sep="\n")
        draw.text((10, 10), txt, font=font, fill="#000000")
        cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', cv2_im_processed)
        k = cv2.waitKey(1)
        engine.say(txt)
        engine.runAndWait()
        engine.stop()
        continue
  
vid.release()
cv2.destroyAllWindows()