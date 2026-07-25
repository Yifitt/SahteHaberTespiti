from pathlib import Path

import joblib
import streamlit as st


MODEL_PATH = Path(__file__).parent / "checkpoints" / "anka_dmm_tfidf_logistic_regression.joblib"
REAL_THRESHOLD = 0.20
FAKE_THRESHOLD = 0.34


@st.cache_resource
def load_model():
    """Eğitilmiş TF-IDF + Logistic Regression modelini bir kez yükle."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_claim(model, text: str) -> dict[str, float | str]:
    """Notebook'taki predict_claim akışını arayüz için döndürülebilir hale getir."""
    text = str(text).strip()
    if not text:
        raise ValueError("Boş metin verilemez.")

    probabilities = model.predict_proba([text])[0]
    fake_prob = float(probabilities[1])
    real_prob = float(probabilities[0])

    if fake_prob >= FAKE_THRESHOLD:
        label = "DMM-benzeri yanlış iddia"
    elif fake_prob <= REAL_THRESHOLD:
        label = "ANKA-benzeri gerçek haber"
    else:
        label = "Belirsiz"

    return {
        "label": label,
        "real_probability": real_prob,
        "false_claim_probability": fake_prob,
    }


st.set_page_config(page_title="Sahte Haber Tespiti", page_icon="📰")
st.title("Sahte Haber Tespiti")
st.write(
    "Haber metnini girin. Model, metnin ANKA-benzeri gerçek haber mi "
    "yoksa DMM-benzeri yanlış iddia mı olduğuna dair olasılık üretir."
)

text = st.text_area(
    "Haber metni",
    height=220,
    placeholder="Haber metnini buraya yapıştırın...",
)

if st.button("Analiz et", type="primary"):
    if not text.strip():
        st.warning("Lütfen analiz edilecek bir haber metni girin.")
    else:
        try:
            result = predict_claim(load_model(), text)
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(f"Tahmin yapılırken bir hata oluştu: {error}")
        else:
            st.subheader("Sonuç")
            if result["label"] == "ANKA-benzeri gerçek haber":
                st.success(result["label"])
            elif result["label"] == "DMM-benzeri yanlış iddia":
                st.error(result["label"])
            else:
                st.info(result["label"])

            col1, col2 = st.columns(2)
            col1.metric(
                "ANKA-benzeri olasılık",
                f"%{result['real_probability'] * 100:.2f}",
            )
            col2.metric(
                "DMM-benzeri yanlış iddia olasılığı",
                f"%{result['false_claim_probability'] * 100:.2f}",
            )

            st.caption(
                "Eşikler: ≤ %20 gerçek haber, ≥ %55 yanlış iddia; "
                "aradaki değerler belirsizdir. Bu model dış kaynak doğrulaması yapmaz."
            )
