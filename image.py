import streamlit as st
from io import BytesIO
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

def get_answer(image, text):
    try:
        img = Image.open(BytesIO(image)).convert("RGB")
        inputs = processor(img, text, return_tensors="pt")
        outputs = model.generate(**inputs)
        answer = processor.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return str(e)

def image_process():
    st.header("CHAT WITH IMAGES 🤖")

    with st.columns(1)[0]:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            st.image(uploaded_file, use_column_width=True)

    question = st.text_input("Ask a Question from the IMAGE", key="user_question")

    if uploaded_file and question:
        image = Image.open(uploaded_file)
        image_byte_array = BytesIO()
        image.save(image_byte_array, format='JPEG')
        image_bytes = image_byte_array.getvalue()
        answer = get_answer(image_bytes, question)
        st.success(answer)