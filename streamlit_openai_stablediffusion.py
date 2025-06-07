import streamlit as st
from PIL import Image
import io
import base64
import requests
from openai import OpenAI

# 環境変数からAPIキーを読み込む
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

st.set_page_config(page_title="ジブリ風変換", layout="centered")
st.title("🌸 ジブリ風人物画像変換アプリ")

uploaded_file = st.file_uploader("人物画像をアップロードしてください（PNGまたはJPG）", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="元の画像", use_column_width=True)

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()

    st.info("ステップ1: GPT-4oでプロンプトを生成中...")

    # Step 1: GPT-4oに画像からプロンプト生成させる
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[
            {"role": "system", "content": "あなたは画像プロンプトのエキスパートです。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "この人物の特徴をもとに、スタジオジブリ風の絵画を生成するための英語のプロンプトを作成してください。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ],
        max_tokens=300
    )

    ghibli_prompt = response.choices[0].message.content.strip()
    st.success("生成されたプロンプト:")
    st.code(ghibli_prompt)

    st.info("ステップ2: Stable Diffusionで画像生成中...")

    # Step 2: Replicate API でジブリ風画像生成
    replicate_url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    model_version = "cjwbw/stable-diffusion-v1-4"  # または別のSDジブリ風モデル

    # 画像をReplicateに送る場合、まず画像をimgurなどにアップロードする必要がある
    # 今回は画像をそのままpromptとして使用
    data = {
        "version": model_version,
        "input": {
            "prompt": f"{ghibli_prompt}, studio ghibli style, dreamy, anime, painterly",
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
    }

    prediction = requests.post(replicate_url, json=data, headers=headers)
    prediction_id = prediction.json().get("id")

    if prediction_id:
        import time
        status = "starting"
        while status != "succeeded":
            time.sleep(1)
            poll = requests.get(f"{replicate_url}/{prediction_id}", headers=headers).json()
            status = poll["status"]
            if status == "succeeded":
                output_url = poll["output"][0]
                st.image(output_url, caption="ジブリ風に変換された画像", use_column_width=True)
            elif status == "failed":
                st.error("画像生成に失敗しました。")
                break
    else:
        st.error("Replicate APIからの応答が無効です。")