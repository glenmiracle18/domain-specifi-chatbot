import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_id = "facebook/opt-350m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")
model.eval()

def generate_response(prompt):
    formatted_prompt = f"### Question:\n{prompt}\n\n### Answer:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.8,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded.split("### Answer:")[-1].strip()

def on_submit(prompt_text):
    status_text = "⏳ Generating answer, please wait..."
    # You can return status_text immediately if you use async, but here we just update together
    result = generate_response(prompt_text)
    status_text = ""  # clear after done
    return result, status_text

with gr.Blocks(css="""
.gradio-container { max-width: 720px; margin: auto; font-family: 'Segoe UI', sans-serif; }
#submit-btn { background: #0e76a8; color: white; border-radius: 8px; padding: 12px 24px; }
#submit-btn:hover { background: #095b81; }
footer { display: none !important; }
""") as demo:

    gr.Markdown("""
    <h2 style="text-align:center;">📊 FinGPT Assistant</h2>
    <p style="text-align:center;">Ask a finance or economics question and get a concise AI-powered answer.</p>
    """)

    with gr.Row():
        prompt = gr.Textbox(
            label="Your Question",
            placeholder="e.g. How does inflation affect consumer spending?",
            lines=3
        )

    with gr.Row():
        submit = gr.Button("Generate Answer", elem_id="submit-btn")

    with gr.Row():
        output = gr.Textbox(label="AI Answer", lines=6, interactive=False)

    with gr.Row():
        status = gr.Markdown("")

    submit.click(on_submit, inputs=prompt, outputs=[output, status], queue=True)

demo.launch(share=True)

