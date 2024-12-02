import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"  # Adjust if hosting the FastAPI app elsewhere

st.title("Prompt Analysis Dashboard")

# User Input
prompt = st.text_area("Enter your prompt below:", "")
response_text = None  # Placeholder for the generated response

if st.button("Analyze Prompt"):
    # Generate Response
    st.subheader("Generated Response")
    try:
        response = requests.post(f"{API_BASE}/generate-response", json={"prompt": prompt})
        if response.status_code == 200:
            response_text = response.json()["response"]
            st.write(response_text)
        else:
            st.error(f"Error in generating response: {response.json()}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

    # Confidence Metric
    st.subheader("Confidence Metric")
    try:
        confidence = requests.post(f"{API_BASE}/get-confidence", json={"prompt": prompt})
        if confidence.status_code == 200:
            st.write(f"Confidence: {confidence.json()['confidence']:.4f}")
        else:
            st.error(f"Error in confidence analysis: {confidence.json()}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

    # Entropy Metric
    st.subheader("Entropy Metric")
    try:
        entropy = requests.post(f"{API_BASE}/get-entropy", json={"prompt": prompt})
        if entropy.status_code == 200:
            st.write(f"Entropy: {entropy.json()['entropy']:.4f}")
        else:
            st.error(f"Error in entropy analysis: {entropy.json()}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")

    # QA Reversibility
    if response_text:
        st.subheader("QA Reversibility")
        try:
            reversibility = requests.post(
                f"{API_BASE}/qa-reversibility", json={"prompt": prompt, "response": response_text}
            )
            if reversibility.status_code == 200:
                similarity = reversibility.json()["similarity"]
                st.write(f"Reversibility Similarity: {similarity:.4f}")
            else:
                st.error(f"Error in QA reversibility analysis: {reversibility.json()}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")

    # Token Dynamics
    st.subheader("Token Dynamics")
    try:
        token_dynamics = requests.post(f"{API_BASE}/track-token-dynamics", json={"prompt": prompt})
        if token_dynamics.status_code == 200:
            data = token_dynamics.json()
            tokens = data["tokens"]
            probabilities = data["probabilities"]
            entropies = data["entropies"]

            # Plot Token Dynamics
            import matplotlib.pyplot as plt

            x = list(range(len(tokens)))
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(x, probabilities, 'b-o', label='True Token Probability')
            ax1.set_xlabel('Token Index')
            ax1.set_ylabel('True Token Probability', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1.set_xticks(x)
            ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)

            ax2 = ax1.twinx()
            ax2.plot(x, entropies, 'r--s', label='Entropy')
            ax2.set_ylabel('Entropy', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            plt.title("Token Influence on Model Predictions")
            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)
        else:
            st.error(f"Error in token dynamics analysis: {token_dynamics.json()}")
    except Exception as e:
        st.error(f"Failed to connect to API: {e}")
