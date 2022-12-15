import gradio as gr
from fastai.vision.all import *

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Anime official image Classifier"
description = "<p style='text-align: center'>Identifier si une image provenant d'un anime est officielle ou un fan art. <br/> Peut par exemple aider à la diffusion de contenu en ligne (articles, posts sur les réseaux sociaux) avec des illustrations officielles.</p>"

demo = gr.Interface(fn=predict, title=title, description=description, examples = ['anime-official.jpg', 'anime-fan-art.jpg'], inputs=gr.Image(shape=(512, 512)), outputs=gr.Label(num_top_classes=2))

demo.launch()