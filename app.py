import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# ======================
# Model Loading
# ======================
@st.cache_resource
def load_bilstm():
    # Load full model (architecture + weights + TextVectorization)
    model = tf.keras.models.load_model("models/bilstm_spam_model.h5")
    return model

model = load_bilstm()

# ======================
# Input Preparation
# ======================
def prepare_input(text: str):
    """
    Convert a single string into the correct tensor shape & dtype for the model.
    Handles both (None,) and (None, 1) input shapes.
    """
    # Always start with a tf.string tensor of shape (1,)
    tensor = tf.constant([text], dtype=tf.string)

    # Try to read input shape robustly
    try:
        ishape = model.input_shape  # e.g., (None,) or (None, 1)
    except Exception:
        # Fallback if input_shape is not available (rare)
        ishape = tuple(model.inputs[0].shape)

    # If model expects (None, 1), expand the last dimension
    if isinstance(ishape, (list, tuple)) and len(ishape) == 2 and ishape[1] == 1:
        tensor = tf.expand_dims(tensor, axis=-1)  # (1, 1)

    return tensor

# ======================
# UI
# ======================
st.title("📩 SMS Spam Detector (BiLSTM)")
st.markdown(
    "Classify SMS messages as **Spam 🚨** or **Ham ✅** using a BiLSTM model with built-in preprocessing."
)

with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01,
                          help="Predicted probability above this is labeled Spam.")
    st.caption("Tip: Increase threshold to reduce false positives.")

user_text = st.text_area("✍️ Enter your SMS message:", height=140, placeholder="e.g., WINNER!! You’ve been selected for a free prize…")

col1, col2 = st.columns([1, 1], vertical_alignment="center")
with col1:
    predict = st.button("🔍 Predict", use_container_width=True)
with col2:
    clear = st.button("🧹 Clear", use_container_width=True)

if clear:
    st.session_state.clear()
    st.rerun()

if predict:
    if not user_text or not user_text.strip():
        st.warning("Please enter a message before predicting.")
    else:
        try:
            x = prepare_input(user_text.strip())
            # Silent prediction
            pred = model.predict(x, verbose=0)

            # Support models that may return [[p]] or [p]
            prob = float(pred[0][0]) if np.ndim(pred) == 2 else float(pred[0])

            is_spam = prob >= threshold
            label = "🚨 Spam" if is_spam else "✅ Ham"
            conf = prob if is_spam else (1.0 - prob)

            st.markdown("---")
            st.subheader("🔎 Prediction Result")
            if is_spam:
                st.error(f"**{label}**  •  Confidence: **{conf:.2%}**")
            else:
                st.success(f"**{label}**  •  Confidence: **{conf:.2%}**")

            with st.expander("View details"):
                st.write(
                    {
                        "raw_probability_spam": round(prob, 6),
                        "threshold": threshold,
                        "decision": label.replace("**", ""),
                        "model_input_shape": getattr(model, "input_shape", "unknown"),
                    }
                )

        except Exception as e:
            st.markdown("---")
            st.error("Prediction failed. See details below:")
            st.exception(e)
            st.info(
                "If the error mentions input dtype/shape, ensure your saved model "
                "includes the TextVectorization layer and that you're loading the same file: "
                "`models/bilstm_spam_model.h5`."
            )

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + TensorFlow BiLSTM")
