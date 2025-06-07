import streamlit as st
from PIL import Image
import io
import base64
from openai import OpenAI

# OpenAI APIキーの読み込み
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="ジブリ風画像変換", layout="centered")
st.title("🌸 ジブリ風人物画像変換アプリ（GPT-4o + DALL·E 3）")

# ユーザーから画像アップロード
uploaded_file = st.file_uploader("人物画像をアップロードしてください（PNGまたはJPG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 元画像の表示
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", use_column_width=True)

    # 画像をbase64形式に変換
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.info("ステップ1: GPT-4oがDALL·E 3用の詳細プロンプトを生成中...")

    # GPT-4oでDALL·E 3向けの詳細プロンプトを生成
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {
                "role": "system",
                "content": "You are an expert in generating prompts for anime-style illustrations using DALL·E 3.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Using the visual characteristics of this person, generate a detailed English prompt for DALL·E 3 to create a Studio Ghibli-style anime illustration.

Include:
- Age, gender, facial expression, hairstyle, clothing, and pose
- Background setting (e.g. meadow, forest, village)
- Style keywords like "studio ghibli style", "anime painting", "soft light", "pastel colors"
Return only the prompt text, no explanations.
""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        },
                    },
                ],
            },
        ],
        max_tokens=400,
    )

    # プロンプトを抽出
    ghibli_prompt = response.choices[0].message.content.strip()
    st.success("生成されたプロンプト:")
    st.code(ghibli_prompt)

    st.info("ステップ2: DALL·E 3で画像を生成中...")

    # DALL·E 3で画像を生成
    dalle_response = client.images.generate(
        model="dall-e-3",
        prompt=f"{ghibli_prompt}",
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = dalle_response.data[0].url
    st.image(image_url, caption="ジブリ風に変換された画像（DALL·E 3）", use_column_width=True)
