import gradio as gr
import sys
from pathlib import Path
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import spaces

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Load your existing system
from final_28_09_2025 import WindowsOptimizedQwen3B, RobustCodeExtractor

class EvoAIInterface:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the Qwen model with optimized settings for cloud"""
        try:
            model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Model loading error: {e}")
    
    @spaces.GPU
    def generate_code(self, problem_description, max_length=1024):
        """Generate code using the evolution system"""
        if not self.model:
            return "Model not loaded properly"
        
        try:
            # Use your existing WindowsOptimizedQwen3B logic
            system = WindowsOptimizedQwen3B()
            result = system.generate_code(problem_description)
            return result
        except Exception as e:
            return f"Generation error: {e}"

# Initialize the interface
evo_ai = EvoAIInterface()

# Create Gradio interface
with gr.Blocks(title="EvoAI 3B - Code Evolution System") as demo:
    gr.Markdown("# 🚀 EvoAI 3B Code Evolution System")
    gr.Markdown("Advanced code generation using Qwen 2.5 Coder-3B with constitutional AI and evolution loops")
    
    with gr.Row():
        with gr.Column():
            problem_input = gr.Textbox(
                label="Problem Description",
                placeholder="Describe the coding problem you want to solve...",
                lines=5
            )
            max_length = gr.Slider(
                minimum=256,
                maximum=2048,
                value=1024,
                step=128,
                label="Max Generation Length"
            )
            generate_btn = gr.Button("🔄 Generate Code", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="Generated Code",
                lines=20,
                show_copy_button=True
            )
    
    generate_btn.click(
        fn=evo_ai.generate_code,
        inputs=[problem_input, max_length],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
