First I want to understand this code. My understanding is as below,
- download the CLIP model from OpenAI
- create a wrapper with export_onnx.py and export the model to ONNX format
- compile it to run on "Qualcomm XR2 Gen 2" 
- Then run inference with sample data in inference.py and measure recall value.

## questions
1) It looks like model is downloaded from qai_hub_models. Not open AI clip implementation
    from qai_hub_models.models.openai_clip.model import OpenAIClip
    What does OpenAIClip do?
    In the process, where OpenAI Clip is being used?

2) Compiling & Running inference on target platform takes time.
    - Can we write inference_local.py which can import model from clip_model module, run inference on sample data?
    - Once we iterate & get hold of the model and code locally, we can push it to run on target?

3) Can you explain step by step code flow, I followed ReadME.md file below steps, but I want to understand each step.
    python export_onnx.py
    python compile_and_profile.py
    python upload_dataset.py
    python inference.py

