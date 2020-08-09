import time
from script_buddy.utils import load_model, generate
import json
import math
import os
import PIL
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary

import streamlit as st
@st.cache(allow_output_mutation=True)

def generate_media(sample):
    font = ImageFont.truetype("fonts/cour.ttf", 60)
    img = Image.open(os.path.join("images","paper.png"))

    d = ImageDraw.Draw(img)


    d.text((0,0), sample,(0,0,0),font= font,quality =20)

    img.save(os.path.join('images','script_output.jpeg'),quality =90, optimize= True)


checkpoint_path = "./model/model.ckpt-3000000"
vocab_file = "./im2txt/data/word_counts.txt"
input_file = "./images/0.jpg"
def loader():
    return load_model() 


def main():
    st.title("home work")
    st.write("""
    This is a scrit generator
    ***
    """)
    max_length = st.sidebar.slider(
    """ Max Script Length 
    (Longer length, slower generation)""",
    50,
    1000
    )
    if st.sidebar.button("Generate"):
        g = tf.Graph()
        with g.as_default():
            model = inference_wrapper.InferenceWrapper()
            restore_fn = model.build_graph_from_config(configuration.ModelConfig(), checkpoint_path)

        vocab = vocabulary.Vocabulary(vocab_file)
        sentence=" "
        with tf.Session(graph=g) as sess:

            restore_fn(sess)
            generator = caption_generator.CaptionGenerator(model, vocab)
            image = tf.gfile.FastGFile(input_file, 'rb').read() 
            img_placeholder=st.empty()
            captions = generator.beam_search(sess, image)
            for i, caption in enumerate(captions):
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                sentence = " ".join(sentence)
                if i == 0:
                    title = sentence
                    st.text(sentence)
                    break
            img_placeholder.image(image,caption=sentence,use_column_width=True)  

        sess.close()

    
        context=sentence
        model, tokenizer = loader()
        sample = generate(model,tokenizer,input_text=context,max_length=max_length)
        st.text(sample[0])
main()

