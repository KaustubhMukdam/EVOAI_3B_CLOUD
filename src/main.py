import argparse
import sys
import os
import torch
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.rtx3050_optimized_loop import RTX3050OptimizedLoop

def setup_rtx3050_optimizations():
    """Setup optimizations specific to RTX 3050 6GB"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 for better performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Clear cache and set memory fraction
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.90)  # Use 90% of 6GB
        
        # Get GPU info
        device_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"GPU Optimizations Enabled: {device_name}")
        print(f"Total VRAM: {total_memory:.1f}GB")
        print(f"Memory fraction set to 90%: {total_memory * 0.9:.1f}GB available")
        
        return True
    else:
        print("CUDA not available - running on CPU (not recommended for 3B model)")
        return False

def main():
    parser = argparse.ArgumentParser(description="EvoAI 3B Parameter Model - RTX 3050 Optimized")
    parser.add_argument("--model-config", default="config/model_3b_rtx3050.yaml", 
                       help="Model configuration file")
    parser.add_argument("--evolution-config", default="config/evolution_3b.yaml",
                       help="Evolution configuration file")
    parser.add_argument("--generations", type=int, default=3,
                       help="Number of evolution generations to run")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Setup GPU optimizations
    gpu_available = setup_rtx3050_optimizations()
    
    if not gpu_available:
        print("Warning: GPU not available. 3B model requires significant computational resources.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    print(f"Starting EvoAI 3B Parameter Evolution")
    print(f"Model config: {args.model_config}")
    print(f"Evolution config: {args.evolution_config}")
    print(f"Generations: {args.generations}")
    
    try:
        # Initialize and run evolution loop
        evolution_loop = RTX3050OptimizedLoop(
            model_config=args.model_config,
            evolution_config=args.evolution_config
        )
        
        evolution_loop.run_evolution(max_generations=args.generations)
        
        print("\nEvolution completed successfully!")
        print("Check 'experiments/results/' for detailed results")
        
    except KeyboardInterrupt:
        print("\nEvolution interrupted by user")
    except Exception as e:
        print(f"\nError during evolution: {e}")
        raise

if __name__ == "__main__":
    main()
