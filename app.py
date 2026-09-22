"""A small local multimodal mushroom chatbot."""

import json
import os
import platform
from pathlib import Path
from threading import Thread

# Let unsupported MPS operations fall back to the CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gradio as gr
import torch
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TextIteratorStreamer,
    logging,
)


MODEL_NAME = "Qwen/Qwen3.5-2B"
DEVICE = "mps" if platform.system() == "Darwin" else "auto"
APP_DIR = Path(__file__).resolve().parent
INSTRUCTIONS = (APP_DIR / "instructions.txt").read_text(encoding="utf-8")

# Serve only the assets folder so the background can be used by the page CSS.
gr.set_static_paths(paths=[APP_DIR / "assets"])

FOREST_THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="emerald",
    neutral_hue="stone",
).set(
    button_primary_background_fill="#245c3a",
    button_primary_background_fill_hover="#19452b",
    button_primary_border_color="#245c3a",
    button_primary_border_color_hover="#19452b",
    loader_color="#245c3a",
)

# A strong overlay keeps the photograph subtle and the chat easy to read.
BACKGROUND_CSS = """
.gradio-container {
    background-image:
        linear-gradient(rgba(244, 249, 244, 0.92), rgba(235, 244, 237, 0.92)),
        url('/gradio_api/file=assets/background.jpeg');
    background-position: center;
    background-size: cover;
    background-attachment: fixed;
}
.dark.gradio-container, .dark .gradio-container {
    background-image:
        linear-gradient(rgba(8, 25, 15, 0.90), rgba(12, 34, 21, 0.90)),
        url('/gradio_api/file=assets/background.jpeg');
}
"""

# The optional fast kernels mentioned by Transformers require CUDA. The normal
# PyTorch fallback is correct on Apple Silicon, so hide those startup warnings.
logging.set_verbosity_error()

# Stop before loading weights instead of crashing when macOS cannot provide MPS.
if DEVICE == "mps" and not torch.backends.mps.is_available():
    raise SystemExit(
        "\nERROR: MPS is not available in this PyTorch process.\n"
        "The model weights were not loaded. Close this process and relaunch with:\n"
        "python app.py"
    )

# Use Apple Silicon GPU acceleration on macOS.
model = AutoModelForMultimodalLM.from_pretrained(MODEL_NAME, device_map=DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def prepare(conversation):
    """Convert a conversation into tensors for the model.

    Parameters:
        conversation: Messages in the Hugging Face chat format.

    Returns:
        Tokenized text and processed images on the model device.
    """
    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)


def generate(conversation, max_new_tokens=512):
    """Generate one complete deterministic answer.

    Parameters:
        conversation: Messages in the Hugging Face chat format.
        max_new_tokens: Maximum length of the answer.

    Returns:
        The decoded model answer.
    """
    inputs = prepare(conversation)
    output_ids = model.generate(
        **inputs,
        do_sample=False,  # temperature 0: make repeated predictions more consistent
        max_new_tokens=max_new_tokens,
    )
    answer_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return processor.decode(answer_ids, skip_special_tokens=True).strip()


def stream(conversation):
    """Yield a deterministic answer as text chunks are generated.

    Parameters:
        conversation: Messages in the Hugging Face chat format.

    Yields:
        The next decoded text chunk.
    """
    inputs = prepare(conversation)
    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    errors = []

    def run_model():
        """Run generation in the background so the streamer can be read."""
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                do_sample=False,
                max_new_tokens=1024,
            )
        except Exception as error:  # forward background errors to Gradio
            errors.append(error)
            streamer.end()

    Thread(target=run_model, daemon=True).start()
    yield from streamer
    if errors:
        raise errors[0]


def image_path(upload):
    """Get a filesystem path from a Gradio upload value.

    Parameters:
        upload: A path, dictionary, or Gradio file object.

    Returns:
        The uploaded image path as a string.
    """
    if isinstance(upload, dict):
        return upload["path"]
    return str(getattr(upload, "path", upload))


def clean_image_json(text):
    """Extract and normalize the model's mushroom JSON.

    Parameters:
        text: Raw model output that should contain one JSON object.

    Returns:
        A dictionary with exactly the fields required by the assignment.
    """
    try:
        data = json.loads(text[text.index("{") : text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    try:
        confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0

    visible = data.get("visible", [])
    if not isinstance(visible, list):
        visible = []
    visible = [part for part in visible if part in {"cap", "hymenium", "stipe"}]

    edible = data.get("edible", False)
    if not isinstance(edible, bool):
        edible = str(edible).lower() == "true"

    return {
        "common_name": str(data.get("common_name", "unknown")),
        "genus": str(data.get("genus", "unknown")),
        "confidence": confidence,
        "visible": visible,
        "color": str(data.get("color", "unknown")),
        "edible": edible,
    }


def inspect_image(path):
    """Ask the model for the required structured mushroom description.

    Parameters:
        path: Path to the uploaded image.

    Returns:
        A normalized mushroom description dictionary.
    """
    request = """Analyze this mushroom image. Return only valid JSON with exactly:
{"common_name": string, "genus": string, "confidence": number from 0 to 1,
"visible": array containing only "cap", "hymenium", and/or "stipe",
"color": string, "edible": boolean}.
Use "unknown", a low confidence, and edible=false when the image is insufficient."""
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTIONS}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "path": path},
                {"type": "text", "text": request},
            ],
        },
    ]
    result = clean_image_json(generate(conversation, max_new_tokens=256))
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def chat(message, history, private_history):
    """Handle one Gradio message and stream the visible response.

    Parameters:
        message: Text and an optional image from the multimodal textbox.
        history: Visible Gradio history; used to detect a cleared chat.
        private_history: Model history, including hidden image JSON.

    Yields:
        The partial answer and updated private history.
    """
    # Clearing the visible chat also starts a new private conversation.
    if not history:
        private_history = []
    conversation = list(private_history) or [
        {"role": "system", "content": [{"type": "text", "text": INSTRUCTIONS}]}
    ]

    question = message.get("text", "").strip()
    uploads = message.get("files", [])
    content = []

    if uploads:
        path = image_path(uploads[0])
        image_data = inspect_image(path)
        content.append({"type": "image", "path": path})

        # This text is sent only to the model, not to the visible Gradio history.
        task = (
            f"The private image analysis is: {json.dumps(image_data)}. "
            "Do not show or quote the JSON. "
        )
        if question:
            task += f"Answer the user's question: {question}"
        else:
            task += "Give the user a short natural-language summary of the image."
        content.append({"type": "text", "text": task})
    else:
        content.append({"type": "text", "text": question})

    conversation.append({"role": "user", "content": content})

    answer = ""
    for chunk in stream(conversation):
        answer += chunk
        yield answer, private_history

    conversation.append(
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    )
    yield answer, conversation


with gr.Blocks() as demo:
    private_history = gr.State([])
    gr.ChatInterface(
        fn=chat,
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            file_count="single",
            file_types=["image"],
            placeholder="Ask about mushrooms or upload a mushroom image",
        ),
        additional_inputs=[private_history],
        additional_outputs=[private_history],
        title="Mushroom Chatbot",
        description="Ask the local mushroom expert a question or upload one image.",
        run_examples_on_click=True,
        examples=[
            [{"text": "List the most common mushrooms in Sweden", "files": []}, None],
            [{"text": "List toxic mushrooms", "files": []}, None],
            [
                {"text": "What is the biggest mushroom in the world?", "files": []},
                None,
            ],
            [
                {
                    "text": "What is this?",
                    "files": [str(APP_DIR / "data" / "mushroom_1.jpg")],
                },
                None,
            ],
        ],
    )


if __name__ == "__main__":
    demo.launch(theme=FOREST_THEME, css=BACKGROUND_CSS)
