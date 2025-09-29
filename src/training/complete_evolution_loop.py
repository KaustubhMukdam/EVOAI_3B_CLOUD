# Modified evolution_loop.py with Correct Paths and SWE-bench Integration




import os
import json
import yaml
import time
import logging
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Updated imports with SWE-bench integration
from models.qwen_3b_optimized import OptimizedQwen3B
from models.memory_efficient_evolution import MemoryEfficientEvolution, OptimizedAgent
from benchmarks.swe_bench_runner import SWEBenchRunner  # Updated to use real SWE-bench

class RTX3050OptimizedLoop:
    def __init__(self, model_config: str, evolution_config: str):
        self.model = OptimizedQwen3B(model_config)
        self.evolution_engine = MemoryEfficientEvolution(evolution_config)
        
        # Updated to use real SWE-bench runner with enhanced features
        self.swe_runner = SWEBenchRunner(evolution_config)
        
        with open(evolution_config, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Memory monitoring
        self.memory_check_interval = 10
        self.max_gpu_memory_percent = 95
        
        # Evolution tracking
        self.current_generation = 0
        self.best_performance = 0.0
        
    def setup_logging(self):
        """Setup logging for training progress"""
        log_dir = Path("experiments/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"evoai_3b_rtx3050_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def monitor_memory_usage(self) -> bool:
        """Monitor memory usage and return True if safe to continue"""
        memory_info = self.model.get_memory_usage()
        
        # Log memory status
        self.logger.info(f"Memory Status - GPU: {memory_info.get('gpu_usage_percent', 0):.1f}%, "
                        f"RAM: {memory_info.get('system_ram_percent', 0):.1f}%")
        
        # Check if memory usage is too high
        if memory_info.get('gpu_usage_percent', 0) > self.max_gpu_memory_percent:
            self.logger.warning("High GPU memory usage detected, clearing caches...")
            self.model.clear_memory_cache()
            return False
        
        return True
    
    def generate_solutions_for_agent(self, agent: OptimizedAgent, num_problems: int = 15) -> List[str]:
        """Generate code solutions for an agent with SWE-bench integration and memory management"""
        solutions = []
        
        # Get problems in easy-first order for better success rates
        ordered_problems = self.swe_runner.get_ordered_problems(
            order="easy_first", 
            limit=num_problems
        )
        
        self.logger.info(f"🤖 Generating {len(ordered_problems)} solutions for agent {agent.agent_id}")
        
        for i, problem_data in enumerate(ordered_problems):
            try:
                # Monitor memory before generation
                if not self.monitor_memory_usage():
                    time.sleep(2)  # Brief pause for memory cleanup
                
                # Extract problem information
                problem_statement = problem_data.get("problem_statement", "")
                instance_id = problem_data.get("instance_id", f"problem_{i}")
                repo = problem_data.get("repo", "unknown")
                
                # Create enhanced prompt for SWE-bench GitHub issues
                enhanced_prompt = f"""Fix this GitHub issue by writing executable Python code:

Repository: {repo}
Issue ID: {instance_id}

Problem Description:
{problem_statement}

Requirements:
- Write valid Python code only
- Include necessary imports
- Add error handling
- Fix the specific issue described
- Ensure code is executable

Python solution:"""
                
                # Generate solution based on agent approach
                if agent.approach == "constitutional_enhanced":
                    result = self.model.constitutional_generation(enhanced_prompt, max_revisions=2)
                    solution = result['final_code']
                else:
                    solution = self.model.generate_code(enhanced_prompt, max_new_tokens=300)
                
                solutions.append(solution)
                
                # Progress logging and periodic memory cleanup
                if (i + 1) % 3 == 0:
                    self.logger.info(f"📝 Generated {i+1}/{len(ordered_problems)} solutions")
                    self.model.clear_memory_cache()
                    
            except Exception as e:
                self.logger.error(f"Error generating solution {i} for agent {agent.agent_id}: {e}")
                solutions.append("# Error in generation")
                self.model.clear_memory_cache()
        
        self.logger.info(f"✅ Generated {len(solutions)} solutions with enhanced SWE-bench prompts")
        return solutions
    
    def evaluate_agent_performance(self, agent: OptimizedAgent) -> Dict[str, Any]:
        """Evaluate agent performance with staged testing using real SWE-bench"""
        agent_id = agent.agent_id
        self.logger.info(f"🧪 Evaluating agent {agent_id} on SWE-bench")
        
        # Generate solutions
        subset_sizes = self.cfg['evaluation']['subset_sizes']
        max_problems = max(subset_sizes)
        
        solutions = self.generate_solutions_for_agent(agent, max_problems)
        
        # Run staged evaluation with easy-first ordering
        stage_results = {}
        final_performance = 0.0
        promotion_threshold = self.cfg['evaluation']['promotion_threshold']
        
        for stage_size in subset_sizes:
            self.logger.info(f"🎯 Running SWE-bench stage evaluation: {stage_size} problems")
            
            # Use SWE-bench runner's evaluation method
            eval_result = self.swe_runner.run_subset_evaluation(
                solutions[:stage_size], 
                stage_size,
                problem_order="easy_first"  # Use easy problems first
            )
            
            stage_performance = eval_result["pass_rate"]
            stage_results[f'stage_{stage_size}'] = eval_result
            final_performance = stage_performance
            
            self.logger.info(f"📊 SWE-bench Stage {stage_size}: {stage_performance:.1%} pass rate")
            
            # Check promotion threshold
            if stage_performance < promotion_threshold:
                self.logger.info(f"❌ Failed promotion threshold ({promotion_threshold:.1%})")
                break
            
            # Memory cleanup between stages
            self.model.clear_memory_cache()
        
        # Calculate constitutional score
        constitutional_score = self.calculate_constitutional_score(agent, solutions[:5])
        
        # Combined fitness score
        alpha = self.cfg['rewards']['alpha_tests_passed']
        beta = self.cfg['rewards']['beta_constitution_score']
        combined_score = alpha * final_performance + beta * constitutional_score
        
        result = {
            'agent_id': agent.agent_id,
            'performance': final_performance,
            'constitutional_score': constitutional_score,
            'combined_fitness': combined_score,
            'stage_results': stage_results,
            'approach': agent.approach,
            'features': getattr(agent, 'features', []),
            'dataset_name': getattr(self.swe_runner, 'dataset_name', 'unknown')
        }
        
        self.logger.info(f"📈 Agent {agent_id}: Performance={final_performance:.1%}, "
                        f"Constitutional={constitutional_score:.1%}, Combined={combined_score:.3f}")
        
        return result
    
    def calculate_constitutional_score(self, agent: OptimizedAgent, solutions: List[str]) -> float:
        """Calculate constitutional AI compliance score"""
        if not solutions:
            return 0.0
        
        total_score = 0.0
        evaluated_solutions = 0
        
        for i, solution in enumerate(solutions[:3]):  # Evaluate subset for efficiency
            try:
                problem = f"Programming problem {i+1}"
                
                # Use constitutional principles if available
                if hasattr(self.model, 'constitutional_principles') and self.model.constitutional_principles:
                    principle = self.model.constitutional_principles[i % len(self.model.constitutional_principles)]
                else:
                    principle = "Write clean, readable, and secure code"
                
                critique_prompt = f"""Rate code quality 0-10 for principle: "{principle}"
Code: {solution[:200]}...
Rating (0-10):"""
                
                critique = self.model.generate_with_memory_management(critique_prompt, 50)
                
                # Extract numeric score
                score = self.extract_numeric_score(critique)
                total_score += score
                evaluated_solutions += 1
                
            except Exception as e:
                self.logger.error(f"Error in constitutional evaluation: {e}")
                continue
        
        return total_score / max(evaluated_solutions, 1) / 10.0  # Normalize to 0-1
    
    def extract_numeric_score(self, text: str) -> float:
        """Extract numeric score from text response"""
        import re
        numbers = re.findall(r'\\b([0-9]|10)\\b', text)
        if numbers:
            return float(numbers[0])
        return 5.0  # Default middle score
    
    def run_generation(self, generation_idx: int) -> List[Dict[str, Any]]:
        """Run a single generation with memory optimization"""
        self.current_generation = generation_idx
        
        self.logger.info(f"\\n{'='*60}")
        self.logger.info(f"🧬 GENERATION {generation_idx}")
        self.logger.info(f"{'='*60}")
        
        generation_start_time = time.time()
        
        # Get population for this generation
        population = self.evolution_engine.next_generation()
        self.logger.info(f"👥 Population size: {len(population)}")
        
        # Evaluate each agent
        evaluation_results = []
        for i, agent in enumerate(population):
            agent_start_time = time.time()
            
            self.logger.info(f"\\n🤖 Evaluating Agent {i+1}/{len(population)}: {agent.agent_id}")
            
            try:
                result = self.evaluate_agent_performance(agent)
                
                # Update agent with results
                agent.performance = result['performance']
                agent.constitutional_score = result['constitutional_score']
                
                evaluation_results.append(result)
                
                agent_time = time.time() - agent_start_time
                self.logger.info(f"⏱️ Agent evaluation completed in {agent_time:.1f}s")
                
            except Exception as e:
                self.logger.error(f"❌ Error evaluating agent {agent.agent_id}: {e}")
                # Create dummy result for failed agent
                result = {
                    'agent_id': agent.agent_id,
                    'performance': 0.0,
                    'constitutional_score': 0.0,
                    'combined_fitness': 0.0,
                    'error': str(e)
                }
                evaluation_results.append(result)
            
            # Memory cleanup after each agent
            self.model.clear_memory_cache()
            
            # Brief pause to prevent overheating
            time.sleep(1)
        
        # Update evolution archive
        self.evolution_engine.update_archive_efficient(population)
        
        # Track best performance
        if evaluation_results:
            generation_best = max(evaluation_results, key=lambda x: x["performance"])
            if generation_best["performance"] > self.best_performance:
                self.best_performance = generation_best["performance"]
                self.logger.info(f"🏆 NEW BEST PERFORMANCE: {self.best_performance:.1%}")
                self.save_best_agent(generation_best, generation_idx)
        
        # Save generation results
        generation_time = time.time() - generation_start_time
        self.save_generation_results(generation_idx, evaluation_results, generation_time)
        
        self.logger.info(f"⏱️ Generation {generation_idx} completed in {generation_time/60:.1f} minutes")
        
        return evaluation_results
    
    def save_best_agent(self, best_agent: Dict[str, Any], generation: int):
        """Save the best performing agent"""
        save_path = Path("models/evolved_models")
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_agent_gen{generation:02d}_{timestamp}.json"
        
        with open(save_path / filename, "w") as f:
            json.dump(best_agent, f, indent=2, default=str)
        
        self.logger.info(f"💾 Best agent saved: {filename}")
    
    def save_generation_results(self, generation_idx: int, results: List[Dict[str, Any]], generation_time: float):
        """Save generation results to file with enhanced information"""
        results_dir = Path("experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate generation statistics
        performances = [r["performance"] for r in results if "performance" in r]
        constitutional_scores = [r["constitutional_score"] for r in results if "constitutional_score" in r]
        
        generation_stats = {
            "max_performance": max(performances) if performances else 0.0,
            "mean_performance": sum(performances) / len(performances) if performances else 0.0,
            "max_constitutional": max(constitutional_scores) if constitutional_scores else 0.0,
            "mean_constitutional": sum(constitutional_scores) / len(constitutional_scores) if constitutional_scores else 0.0
        }
        
        output_data = {
            'generation': generation_idx,
            'timestamp': datetime.now().isoformat(),
            'generation_time_seconds': generation_time,
            'population_size': len(results),
            'results': results,
            'statistics': generation_stats,
            'memory_status': self.evolution_engine.get_memory_status(),
            'model_memory': self.model.get_memory_usage(),
            'dataset_name': getattr(self.swe_runner, 'dataset_name', 'unknown')
        }
        
        filename = f"generation_{generation_idx:03d}.json"
        with open(results_dir / filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        self.logger.info(f"📊 Generation results saved: {filename}")
    
    def run_evolution(self, max_generations: int = 3):
        """Run the complete evolution process optimized for RTX 3050 with SWE-bench"""
        self.logger.info(f"🚀 Starting EvoAI 3B optimization on RTX 3050 - {max_generations} generations")
        self.logger.info(f"📊 Using dataset: {getattr(self.swe_runner, 'dataset_name', 'unknown')}")
        
        start_time = time.time()
        
        try:
            for generation in range(1, max_generations + 1):
                gen_start_time = time.time()
                
                # Run generation
                results = self.run_generation(generation)
                
                # Calculate generation statistics
                performances = [r['performance'] for r in results if 'performance' in r]
                best_performance = max(performances) if performances else 0.0
                avg_performance = sum(performances) / len(performances) if performances else 0.0
                
                gen_time = time.time() - gen_start_time
                
                self.logger.info(f"✅ Generation {generation} completed in {gen_time/60:.1f} minutes")
                self.logger.info(f"🏆 Best performance: {best_performance:.1%}, Average: {avg_performance:.1%}")
                
                # Memory status report
                memory_status = self.evolution_engine.get_memory_status()
                self.logger.info(f"📈 Archive size: {memory_status['archive_size']}, "
                               f"GPU memory: {memory_status.get('gpu_memory_allocated_mb', 0):.0f}MB")
                
                # Cleanup between generations
                self.model.clear_memory_cache()
                time.sleep(2)
        
        except KeyboardInterrupt:
            self.logger.info("⏹️ Evolution interrupted by user")
        except Exception as e:
            self.logger.error(f"❌ Error in evolution loop: {e}", exc_info=True)
        
        total_time = time.time() - start_time
        
        self.logger.info(f"\\n{'='*60}")
        self.logger.info(f"🏁 EVOLUTION COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"🏆 Best performance achieved: {self.best_performance:.1%}")
        self.logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"📊 Generations completed: {self.current_generation}")
        
        # Save final results
        self.save_final_results()
    
    def save_final_results(self):
        """Save final evolution results and best models with enhanced information"""
        results_dir = Path("experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Get best agents from archive
        best_agents = sorted(self.evolution_engine.archive, 
                           key=lambda x: x.performance + 0.3 * x.constitutional_score, 
                           reverse=True)[:5]
        
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'total_generations': self.evolution_engine.generation,
            'archive_size': len(self.evolution_engine.archive),
            'best_overall_performance': self.best_performance,
            'dataset_name': getattr(self.swe_runner, 'dataset_name', 'unknown'),
            'best_agents': [
                {
                    'agent_id': agent.agent_id,
                    'performance': agent.performance,
                    'constitutional_score': agent.constitutional_score,
                    'features': getattr(agent, 'features', []),
                    'approach': agent.approach,
                    'generation': agent.generation
                }
                for agent in best_agents
            ],
            'final_memory_status': self.evolution_engine.get_memory_status()
        }
        
        with open(results_dir / "final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Final results saved. Best performance: {best_agents[0].performance:.1%}")
'''```

## Key Changes Made:

✅ **Updated Import**: Changed from `LightweightSWERunner` to `SWEBenchRunner` for real SWE-bench integration

✅ **Enhanced Solution Generation**: 
- Uses `get_ordered_problems()` with easy-first ordering
- Creates GitHub issue-specific prompts with repository context
- Increased `max_new_tokens` to 300 for better solutions

✅ **Real SWE-bench Evaluation**:
- Uses `run_subset_evaluation()` method from SWE-bench runner
- Tracks dataset name and problem ordering
- Enhanced logging with SWE-bench specific information

✅ **Better Performance Tracking**:
- Tracks best performance across generations
- Saves best agents with enhanced metadata
- Includes dataset information in all results

✅ **Memory Management Preserved**:
- All your existing RTX 3050 memory optimizations maintained
- Thermal management and cleanup procedures intact
- Conservative memory usage patterns preserved

✅ **Enhanced Logging**:
- More detailed progress reporting
- SWE-bench specific status messages
- Better error handling and reporting

This modified version maintains all your existing optimizations while integrating the real SWE-bench dataset with smart problem ordering for better performance on your RTX 3050 setup.'''