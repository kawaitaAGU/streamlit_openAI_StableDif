import streamlit as st
from PIL import Image
import io
import base64
from openai import OpenAI

# OpenAIクライアント設定（ChatGPT Plus ユーザー向け）
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="ジブリ風変換", layout="centered")
st.title("🌸 GPT-4o + DALL·E 3 によるジブリ風画像変換")

uploaded_file = st.file_uploader("人物画像をアップロードしてください（PNGまたはJPG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", use_column_width=True)

    # Base64形式に変換
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.info("ステップ1: GPT-4oがジブリ風プロンプトを生成中...")

    # GPT-4oでプロンプト生成
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "あなたはプロの画像生成プロンプト作成者です。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この人物の特徴をもとに、スタジオジブリ風の絵画を生成するための英語プロンプトを作成してください。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=300
    )

    ghibli_prompt = response.choices[0].message.content.strip()
    st.success("生成されたプロンプト（DALL·E 3用）:")
    st.code(ghibli_prompt)

    st.info("ステップ2: DALL·E 3でジブリ風画像を生成中...")

    # DALL·E 3による画像生成
    dalle_response = client.images.generate(
        model="dall-e-3",
        prompt=f"{ghibli_prompt}, studio ghibli style, anime painting, soft background",
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = dalle_response.data[0].url
    st.image(image_url, caption="DALL·E 3 によるジブリ風画像", use_column_width=True)
