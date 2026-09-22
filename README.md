# Assignment: Building a Mushroom Chatbot
After the completion of the Lab, you are welcome to start working on the assignment. Here, you will be developing the mushroom chatbot presented during the lab seminar.

Start by downloading all the material: assignment.zip

The goal of the chatbot is to be able to answer questions about mushrooms and process images of mushrooms. The chatbot should be able to provide information about images of mushrooms and answer questions about them.

Write a report, using **[this latex template](https://www.overleaf.com/read/yzcrrckvrwgy#143b01)**, that contains the answers to the questions in `Things to check` and the code for the exercises in `Main exercises`. The report should also contain examples of the chatbot responses for the exercises. Submit the report, alongside your code, through Canvas. **The report should not exceed three pages**, excluding the appendix.

#### Main exercises:
 1. Create a simple chatbot with a graphics interface.
 2. Make the chatbot multimodal so that it can read mushroom images.
 3. Add instructions to the chatbot, so that it behaves like a mushroom expert, and that it stirs the conversation in the right direction (i.e. mushrooms related).
 4. Stream the answers and display the chunks of text as they are being processed.
 5. When the chatbot receives an image, ask it to produce a JSON response with the following fields: common_name, genus, confidence (of the prediction), visible (what parts of the mushrooms are visible in the image, from the set {cap, hymenium, stipe}, the color of the mushroom in the picture and the edibility of the mushroom. As an example:
      ```json
      {
      "common_name": "Inkcap",
      "genus": "Coprinus",
      "confidence": 0.5,
      "visible": ["cap", "hymenium", "stipe"],
      "color": "orange",
      "edible": true
      }
      ```
    Do not display the information in the chatbot, but print it in the console. Let the chatbot keep this information in mind for the next questions.
 6. If the user asked a question when they sent the image, after processing said image, answer the question, but if no questions were asked, provide a summary of the processed image, using the information in the JSON response.

#### Things to check (after completing the exercises):
 1. Assess if the model can correctly classify mushrooms? (The value is in the quality of your assessment, not the result itself.) Note: the model cannot classify the parasol mushroom as all information is not visible in the image (the stipe is missing). What happens in this case?
 2. The model is likely not consistent in its predictions (e.g. ask twice to classify the same image). One way is to change the temperature of the model and set it to 0.0 from now on. Note that this solves the problem of reproducibility, and is generally considered a good practice, but doesn't guarantee consistency. What other causes could lead to inconsistent predictions?
 2. Can you find a way to make the chatbot talk about another topic than mushrooms? Is it hard? If not, how would you make it harder?
 3. Can you ask the chatbot to transcribe the text in the `nya_svampboken_p226.jpg` file? Did it do a good job? Are things reinterpreted or missing?
 4. Safety filters:
    * (Gemini) Gemini models come with [safety filters](https://ai.google.dev/gemini-api/docs/safety-settings) that may trigger on certain topics e.g., if asked on how to use mushrooms to hurt someone. In case of many false positives, you can consider lowering the filters. What are the risks and consequences of doing so?
    * (HuggingFace) HF models don't come with safety filters. What are the risks and consequences of that? How would you approach this problem in a risky scenario?
 5. Does the chatbot provide accurate descriptions given the instructions of exercise 5. and 6.? Specifically, is the JSON format correct? Are the fields correctly filled? Are the answers relevant? Does the summary correspond to the JSON response?
 6. Can you check the quality of the chatbot's answers? How would you do that? Try the mushroom_*.jpg images in the `data` folder with various models of various complexities. What are the results? What are the limitations of a mushroom expert chatbot?
 7. The Amanita Muscaria (the red mushroom with white dots) is a famous mushroom. It contains a neurotoxin that can cause delirium among other symptoms, but it is generally not deadly. It can be eaten if prepared correctly. Can you find a way to make the chatbot provide this information? Now, how can you make sure that the chatbot does not provide this information to someone who would eat the mushroom without proper preparation?
 8. Can you find a way to make the chatbot more engaging? How would you do that?

#### Optional exercises:
 1. Add a description to the chat interface.
 2. Add a theme to the chat interface.
 3. Add a few examples to the chat interface.

#### Evaluation

Some of the exercises are more challenging than others and a few even do not have a clear solution. The evaluation will be based on the quality of the implementation, the understanding of the concepts, and the creativity of the assessment in `Things to check`.

Note: The instructions for the use of Generative AI is available in the Home page of the course. Please include a brief paragraph at the end in your submission explaining if and how you used Generative AI.

Resubmissions: The late submission deadline will be 4 weeks after the original deadline. Any submissions after that will be graded in June.

## Run the solution

Use Python 3.10 or newer. The first run downloads the local `Qwen/Qwen3.5-2B`
model (about 4.6 GB), so make sure there is enough disk space and memory.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the local address printed in the terminal. Image JSON is printed in that
terminal but is not displayed in the chat. Edit `instructions.txt` to change the
chatbot's behavior, then restart the app.
