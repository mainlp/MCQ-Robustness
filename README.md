# MCQ-Robustness

<a href="https://huggingface.co/mainlp/MCQ-Classifier-MMLU-XYZ"><img alt="HuggingFace Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-8A2BE2"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2404.08382-b31b1b.svg)](https://arxiv.org/abs/2404.08382)

Repo for paper 

Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think

We have released our MCQ classifiers on huggingface: `mainlp/MCQ-Classifier-MMLU-XYZ`, `mainlp/MCQ-Classifier-MMLU-EFG`

Please refer to their model cards for the details.

## How to use 

Your should construct your input into such format: model_reponse + "\nReferences:" + references + "\nAnswer:"

For example:
```
inputs = " Sure, I'm happy to help! The correct answer is:\n\nB. retraction of the stoma. \nReferences: \nA. high output stomas. \nB. retraction of the stoma. \nC. prolapsed stomas. \nD. herniation around the stoma. \nAnswer:"
```
then feed it to the classifier:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
config = PeftConfig.from_pretrained("mainlp/MCQ-Classifier-MMLU-XYZ")
base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
model = PeftModel.from_pretrained(base_model, "mainlp/MCQ-Classifier-MMLU-XYZ")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
to_classify = f"""<s>[INST] Classify the response.{inputs} [/INST]"""
model_input = tokenizer(to_classify, return_tensors="pt")
output =  merged_model.generate(**model_input, max_new_tokens=1, do_sample=False)
print(tokenizer.decode(output.sequences[0], skip_special_tokens=True))
```

## Cite
```
@article{wang2024look,
  title={Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You Think},
  author={Wang, Xinpeng and Hu, Chengzhi and Ma, Bolei and R{\"o}ttger, Paul and Plank, Barbara},
  journal={arXiv preprint arXiv:2404.08382},
  year={2024}
}
```
