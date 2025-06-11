# **Fine-tuning Mistral-7B using QLoRA for Domain-Specific Text Generation**

## **Project Overview**

This repository hosts a Jupyter Notebook (FineTuning\_Mistral\_7B\_using\_QLoRA.ipynb) that demonstrates the process of fine-tuning the Mistral-7B-v0.1 large language model. The fine-tuning utilizes the QLoRA (Quantized Low-Rank Adaptation) technique on a custom dataset derived from Mary Shelley's "Frankenstein." The goal is to adapt the model's text generation capabilities to the specific stylistic nuances of the novel's prose, showcasing an efficient method for domain-specific LLM adaptation on consumer-grade GPUs.

## **Motivation**

The primary objectives of this project were to:

* Gain practical, hands-on experience with QLoRA fine-tuning, a cutting-edge and memory-efficient approach for adapting large language models.  
* Understand the practical application and integration of key libraries such as Hugging Face transformers and peft for customized LLM development.  
* Perform both quantitative (perplexity) and qualitative (generated text analysis) evaluations to assess the effectiveness of the fine-tuning process.

## **Technologies Used**
```bash
* Python  
* transformers (Hugging Face)  
* peft (Parameter-Efficient Fine-Tuning)  
* bitsandbytes (4-bit quantization)  
* accelerate  
* datasets (Hugging Face)  
* pandas  
* torch (PyTorch)  
* scikit-learn (for data splitting)  
* Jinja2==3.0.3  
* Base Model: mistralai/Mistral-7B-v0.1
```

## **Repository Structure**

* FineTuning\_Mistral\_7B\_using\_QLoRA.ipynb: The main Jupyter Notebook containing all the code for data loading, model setup, QLoRA configuration, training, and evaluation. This file includes executed outputs for easy viewing on GitHub.  
* frankenstein\_chunks.csv: The raw dataset used for fine-tuning, consisting of text chunks extracted from "Frankenstein".  
* README.md: This file, providing a comprehensive overview and instructions for the project.

## **Setup and Installation**

To get this project running on your local machine or a cloud environment like Google Colab, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rakshith-rak69/Finetuning-Mistral7B-QLoRA.git  
   cd Finetuning-Mistral7B-QLoRA
   ```

2. **Create a virtual environment (recommended):**
   ```bash 
   python \-m venv venv  
   source venv/bin/activate  \# On Windows, use \`venv\\Scripts\\activate\`
   ```

3. Install the required packages:  
   Create a requirements.txt file in the root of the repository with the following content, then install:
   ```bash
   pip install \-r requirements.txt
   ```


   **requirements.txt content:**
   ```bash
   bitsandbytes  
   transformers  
   peft  
   accelerate  
   datasets  
   pandas  
   torch  
   scikit-learn  
   Jinja2==3.0.3
   ```

   *Note: A GPU (e.g., NVIDIA CUDA enabled) is highly recommended for efficient training, as fine-tuning large models is computationally intensive.*  
7. Hugging Face Token:  
   The notebook uses huggingface\_hub.login() to access the Mistral-7B-v0.1 model. For security, do not hardcode your Hugging Face API token directly in the notebook when sharing publicly. Instead, the notebook is set up to retrieve the token from an environment variable (HF\_TOKEN) or prompt for interactive login. Ensure your token has read access to models. Replace "YOUR\_HF\_TOKEN\_HERE" with your actual token in the FineTuning\_Mistral\_7B\_using\_QLoRA.ipynb file, or set it as an environment variable HF\_TOKEN.

## **How to Run the Notebook**

1. Open the Notebook:  
   You can open FineTuning\_Mistral\_7B\_using\_QLoRA.ipynb using:  
   * **Google Colab:** Upload the notebook directly to Colab.  
   * **Jupyter Notebook/Lab:** Navigate to the cloned repository in your terminal and run jupyter notebook or jupyter lab.  
   * **VS Code:** Open the folder in VS Code and open the .ipynb file.  
2. **Run Cells Sequentially:** Execute the cells in the notebook from top to bottom.  
   * The first few cells handle package installation and basic setup.  
   * The huggingface\_hub.login() cell will guide you on providing your Hugging Face token.  
   * Subsequent cells will perform data loading, model quantization, tokenization, training, and evaluation.

The notebook provides print statements for progress and final results, including text generations and perplexity calculations.

## **Results**

The fine-tuning process successfully adapted the Mistral-7B model to generate text more in line with the distinctive style and themes of Mary Shelley's "Frankenstein."

### **Perplexity Improvement**

Perplexity, an intrinsic evaluation metric for language models, quantifies how well a probability model predicts a sample. A lower perplexity indicates better model performance, suggesting the model has a stronger understanding of the text's underlying patterns.

* **Base model perplexity:** ```bash tensor(8.5008, device='cuda:0')  ```
* **Finetuned model perplexity:** ```bash tensor(6.2617, device='cuda:0') ```

This reduction in perplexity demonstrates a measurable improvement in the model's ability to predict text in the "Frankenstein" domain, specifically a **\~26.34% decrease** in perplexity.

### **Text Generation Examples**

Below are examples of text generated by the base (untrained) and finetuned models given the same prompt, highlighting the impact of domain-specific fine-tuning:

**Prompt:** "I'm afraid I've created a "

**Base Model Generation:**
```bash
I'm afraid I've created a 2000-level problem with a 100-level solution.

I'm a 2000-level problem.

I'm a 2000-level problem.

I'm a 2000-level problem.

I'm a 2000-level problem.

I'm a 2000-level problem.

I'm a 2
```
**Finetuned Model Generation:**
```bash
I'm afraid I've created a  monster, one whom you are powerless to oppose; and he

will be a constant menace to your peace and happiness.

"I have been driven to these extremes by the remorse I feel for my

crimes.  I have murdered the lovely and the innocent.  I shall be

haunted by the vision of the corpse I have created.  If you consent to

destroy it, I shall be content; but if not,
```

As clearly observed, the finetuned model generates a completion that is contextually relevant to the "Frankenstein" narrative, specifically referencing a "monster" and themes of remorse and creation. This contrasts sharply with the repetitive and generic output from the base model, demonstrating the success of the domain adaptation.

## **Future Enhancements (Optional)**

* **Hyperparameter Tuning:** Experiment with different LoRA parameters (e.g., r, lora\_alpha, lora\_dropout), number of training epochs, and batch sizes to further optimize perplexity and generation quality.  
* **Dataset Expansion/Refinement:** Explore larger or more meticulously curated datasets for potentially greater domain adaptation.  
* **Advanced Evaluation:** Implement more sophisticated human evaluation metrics or task-specific evaluations.  
* **Deployment:** Integrate the finetuned model into a simple web application (e.g., using Streamlit or Gradio) for interactive text generation.

