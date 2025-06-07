import streamlit as st
from PIL import Image
import io
import base64
from openai import OpenAI

# OpenAIクライアント設定
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="ジブリ風画像変換", layout="centered")
st.title("🌸 若々しいジブリ風人物画像変換アプリ（GPT-4o + DALL·E 3）")

# 画像アップローダー
uploaded_file = st.file_uploader("人物画像をアップロードしてください（PNGまたはJPG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", use_column_width=True)

    # base64エンコード
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.info("ステップ1: GPT-4o が DALL·E 3 向けプロンプトを生成中...")

    # GPT-4o による詳細プロンプト生成（若々しい雰囲気を強調）
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

Instructions:
- Make the character look only a little bit younger and a little bit more youthful than the original photo.
- Include details such as age, gender, facial expression, hairstyle, clothing, and pose.
- Describe a vivid, peaceful background (e.g. meadow, forest, village, twilight sky).
- Use stylistic keywords such as "studio ghibli style", "anime painting", "soft light", "pastel colors", "dreamy atmosphere".

Return only the final prompt.
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

    # GPTが生成したプロンプトを表示
    ghibli_prompt = response.choices[0].message.content.strip()
    st.success("生成されたプロンプト（若々しい雰囲気）:")
    st.code(ghibli_prompt)

    st.info("ステップ2: DALL·E 3 による画像生成中...")

    # DALL·E 3 を用いた画像生成
    dalle_response = client.images.generate(
        model="dall-e-3",
        prompt=ghibli_prompt,
        size="1024x1024",
        quality="standard",
        n=1
    )

    image_url = dalle_response.data[0].url
    st.image(image_url, caption="ジブリ風・若々しい人物画像（DALL·E 3）", use_column_width=True)
