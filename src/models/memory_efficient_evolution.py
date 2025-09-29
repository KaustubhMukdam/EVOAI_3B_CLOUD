import torch
import gc
import yaml
import random
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import psutil

@dataclass
class OptimizedAgent:
    agent_id: str
    performance: float
    constitutional_score: float
    features: List[str]
    approach: str
    generation: int
    memory_efficient: bool = True

class MemoryEfficientEvolution:
    def __init__(self, config_path: str = None):
        # Default config if file doesn't exist
        default_config = {
            'evolution': {
                'population_size': 2,
                'archive_size': 10,
                'mutation_rate': 0.15
            },
            'evaluation': {
                'subset_sizes': [3, 5, 8],
                'promotion_threshold': 0.3
            },
            'rewards': {
                'alpha_tests_passed': 0.6,
                'beta_constitution_score': 0.3,
                'gamma_diff_complexity': 0.1
            },
            'memory_optimization': {
                'clear_cache_frequency': 3
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = default_config
        
        self.archive = []
        self.generation = 0
        self.population_size = self.cfg['evolution']['population_size']
        self.archive_size = self.cfg['evolution']['archive_size']
        self.mutation_rate = self.cfg['evolution']['mutation_rate']
        
        # Memory management settings
        self.clear_cache_frequency = self.cfg['memory_optimization']['clear_cache_frequency']
        
    def clear_memory_caches(self):
        """Comprehensive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def create_initial_population(self) -> List[OptimizedAgent]:
        """Create initial population with memory optimization"""
        population = []
        
        approaches = [
            "direct_solution",
            "step_by_step",
            "optimized_approach",
            "constitutional_enhanced"
        ]
        
        for i in range(self.population_size):
            agent = OptimizedAgent(
                agent_id=f"gen{self.generation}_agent{i}",
                performance=0.0,
                constitutional_score=0.0,
                features=random.sample([
                    "error_handling", "optimization", "documentation", 
                    "testing", "modularity", "security"
                ], k=3),
                approach=random.choice(approaches),
                generation=self.generation,
                memory_efficient=True
            )
            population.append(agent)
            
        return population
    
    def tournament_selection(self, population: List[OptimizedAgent], k: int = 2) -> OptimizedAgent:
        """Memory-efficient tournament selection"""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=lambda x: x.performance)
    
    def update_archive_efficient(self, new_agents: List[OptimizedAgent]):
        """Update archive with memory efficiency"""
        # Add successful agents
        for agent in new_agents:
            if agent.performance > 0.2:  # Only keep promising agents
                self.archive.append(agent)
        
        # Sort by combined performance
        self.archive.sort(key=lambda x: x.performance + 0.3 * x.constitutional_score, reverse=True)
        
        # Maintain archive size
        if len(self.archive) > self.archive_size:
            self.archive = self.archive[:self.archive_size]
        
        # Clear memory after archive update
        if len(self.archive) % self.clear_cache_frequency == 0:
            self.clear_memory_caches()
    
    def next_generation(self) -> List[OptimizedAgent]:
        """Generate next population with memory management"""
        if not self.archive:
            return self.create_initial_population()
        
        # For simplicity, create new random population with some elitism
        new_population = []
        
        # Keep best performer (elitism)
        if self.archive:
            best_agent = self.archive[0]
            new_population.append(OptimizedAgent(
                agent_id=f"gen{self.generation + 1}_elite",
                performance=0.0,
                constitutional_score=0.0,
                features=best_agent.features.copy(),
                approach=best_agent.approach,
                generation=self.generation + 1
            ))
        
        # Fill remaining with new agents
        while len(new_population) < self.population_size:
            new_population.extend(self.create_initial_population()[:1])
        
        self.generation += 1
        return new_population[:self.population_size]
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        status = {}
        
        # System memory
        memory = psutil.virtual_memory()
        status['system_memory_percent'] = memory.percent
        status['system_memory_available_gb'] = memory.available / (1024**3)
        
        # GPU memory if available
        if torch.cuda.is_available():
            status['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
            status['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024**2)
            status['gpu_memory_total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        
        # Archive statistics
        status['archive_size'] = len(self.archive)
        status['generation'] = self.generation
        
        return status
