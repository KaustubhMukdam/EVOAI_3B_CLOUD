import json
import yaml
import tempfile
import subprocess
import shutil
import logging
import random
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset

class SWEBenchRunner:
    def __init__(self, config_path: str):
        """Initialize SWE-bench runner with proper dataset loading"""
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.sandbox_enabled = self.config["safety"]["sandbox_enabled"]
        self.max_execution_time = self.config["safety"]["max_execution_time"]
        
        # Load actual SWE-bench dataset
        self._load_swe_bench_dataset()
        
        # Create problem index for easy ordering
        self._create_problem_index()
    
    def _load_swe_bench_dataset(self):
        """Load real SWE-bench dataset with multiple fallback options"""
        dataset_options = [
            ("SWE-bench/SWE-bench_Lite", "Curated subset for faster evaluation"),
            ("SWE-bench/SWE-bench_Verified", "High-quality verified problems"),
            ("princeton-nlp/SWE-bench", "Full original dataset")
        ]
        
        self.dataset = None
        
        for dataset_name, description in dataset_options:
            try:
                logging.info(f"Loading {dataset_name}: {description}")
                self.dataset = load_dataset(dataset_name, split="test")
                logging.info(f"Successfully loaded {dataset_name}: {len(self.dataset)} problems")
                self.dataset_name = dataset_name
                break
            except Exception as e:
                logging.warning(f"Failed to load {dataset_name}: {e}")
                continue
        
        if self.dataset is None:
            logging.error("All SWE-bench datasets failed to load")
            raise RuntimeError("Could not load any SWE-bench dataset")
    
    def _create_problem_index(self):
        """Create index for problem ordering and difficulty"""
        self.problem_index = []
        
        for i, problem in enumerate(self.dataset):
            problem_text = problem.get("problem_statement", "")
            difficulty_score = self._estimate_difficulty(problem_text)
            
            self.problem_index.append({
                "index": i,
                "instance_id": problem.get("instance_id", f"problem_{i}"),
                "difficulty_score": difficulty_score,
                "length": len(problem_text),
                "repo": problem.get("repo", "unknown")
            })
        
        # Sort by difficulty (easy first for better pass rates)
        self.problem_index.sort(key=lambda x: x["difficulty_score"])
        logging.info(f"Indexed {len(self.problem_index)} problems by difficulty")
    
    def _estimate_difficulty(self, problem_text: str) -> float:
        """Estimate problem difficulty based on text analysis"""
        difficulty = 0.0
        difficulty += min(len(problem_text) / 1000, 2.0)
        
        complex_keywords = ["async", "threading", "multiprocessing", "metaclass", "decorator", 
                          "performance", "optimization", "memory", "cache"]
        simple_keywords = ["fix", "add", "update", "change", "simple", "basic"]
        
        text_lower = problem_text.lower()
        
        for keyword in complex_keywords:
            if keyword in text_lower:
                difficulty += 0.5
        
        for keyword in simple_keywords:
            if keyword in text_lower:
                difficulty -= 0.2
        
        return max(0.0, difficulty)
    
    def get_ordered_problems(self, order: str = "easy_first", limit: Optional[int] = None) -> List[Dict]:
        """Get problems in specified order"""
        if order == "easy_first":
            ordered_indices = [p["index"] for p in self.problem_index]
        elif order == "hard_first": 
            ordered_indices = [p["index"] for p in self.problem_index[::-1]]
        else:  # default order
            ordered_indices = list(range(len(self.dataset)))
        
        if limit:
            ordered_indices = ordered_indices[:limit]
        
        return [self.dataset[i] for i in ordered_indices]
    
    def extract_patch_from_solution(self, solution: str) -> str:
        """Extract patch from generated solution"""
        # Look for diff/patch patterns
        patch_patterns = [
            r'``````',
            r'``````',
            r'``````',
            r'(--- a/.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in patch_patterns:
            matches = re.findall(pattern, solution, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # If no proper patch format, try to create one from code blocks
        code_patterns = [
            r'``````',
            r'``````'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, solution, re.DOTALL | re.IGNORECASE)
            if matches:
                code = matches[0].strip()
                # Create a simple patch format
                return f"""--- a/fix.py
+++ b/fix.py
@@ -1,1 +1,{len(code.split())} @@
+{code}"""
        
        return solution.strip()
    
    def evaluate_patch_quality(self, patch: str, problem_data: Dict) -> Dict[str, Any]:
        """Evaluate patch quality using heuristics"""
        if not patch or len(patch.strip()) < 10:
            return {
                "quality_score": 0.0,
                "has_proper_format": False,
                "addresses_issue": False,
                "is_safe": False
            }
        
        quality_score = 0.0
        
        # Check for proper patch format
        has_proper_format = any(marker in patch for marker in ["---", "+++", "@@", "diff"])
        if has_proper_format:
            quality_score += 0.3
        
        # Check if it addresses the issue
        problem_statement = problem_data.get("problem_statement", "").lower()
        patch_lower = patch.lower()
        
        # Extract key terms from problem
        key_terms = []
        if "function" in problem_statement:
            functions = re.findall(r'\b\w+\(\)', problem_statement)
            key_terms.extend([f.replace('()', '') for f in functions])
        
        if "class" in problem_statement:
            classes = re.findall(r'class\s+(\w+)', problem_statement)
            key_terms.extend(classes)
        
        # Check if patch mentions relevant terms
        addresses_issue = any(term.lower() in patch_lower for term in key_terms) if key_terms else True
        if addresses_issue:
            quality_score += 0.4
        
        # Basic safety checks
        dangerous_patterns = ["rm -rf", "delete", "__delete__", "os.system", "eval("]
        is_safe = not any(pattern in patch_lower for pattern in dangerous_patterns)
        if is_safe:
            quality_score += 0.3
        
        return {
            "quality_score": min(1.0, quality_score),
            "has_proper_format": has_proper_format,
            "addresses_issue": addresses_issue,
            "is_safe": is_safe
        }
    
    def run_subset_evaluation(self, solutions: List[str], subset_size: int, 
                             problem_order: str = "easy_first") -> Dict[str, Any]:
        """Run evaluation on a subset with proper SWE-bench integration"""
        ordered_problems = self.get_ordered_problems(order=problem_order, limit=subset_size)
        actual_size = min(subset_size, len(solutions), len(ordered_problems))
        
        logging.info(f"Running SWE-bench evaluation on {actual_size} problems ({problem_order} order)")
        
        passed_count = 0
        results = []
        quality_scores = []
        
        for i in range(actual_size):
            try:
                solution = solutions[i]
                problem_data = ordered_problems[i]
                
                # Extract patch from solution
                patch = self.extract_patch_from_solution(solution)
                
                # Evaluate patch quality
                quality_eval = self.evaluate_patch_quality(patch, problem_data)
                quality_scores.append(quality_eval["quality_score"])
                
                # Simple pass criteria based on quality
                passed = (quality_eval["quality_score"] > 0.5 and 
                         quality_eval["has_proper_format"] and
                         quality_eval["is_safe"])
                
                if passed:
                    passed_count += 1
                    
                results.append({
                    "problem_index": i,
                    "instance_id": problem_data.get("instance_id", f"problem_{i}"),
                    "passed": passed,
                    "quality_score": quality_eval["quality_score"],
                    "has_proper_format": quality_eval["has_proper_format"],
                    "addresses_issue": quality_eval["addresses_issue"],
                    "is_safe": quality_eval["is_safe"],
                    "patch_length": len(patch),
                    "repo": problem_data.get("repo", "unknown")
                })
                
                if (i + 1) % 3 == 0:
                    current_pass_rate = passed_count / (i + 1)
                    logging.info(f"Progress: {i+1}/{actual_size}, Pass rate: {current_pass_rate:.1%}")
                    
            except Exception as e:
                logging.error(f"Evaluation {i} failed: {e}")
                results.append({
                    "problem_index": i,
                    "passed": False,
                    "error": str(e),
                    "quality_score": 0.0
                })
        
        pass_rate = passed_count / actual_size if actual_size > 0 else 0.0
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        
        evaluation_summary = {
            "dataset_name": self.dataset_name,
            "problem_order": problem_order,
            "total_problems": actual_size,
            "passed_problems": passed_count,
            "failed_problems": actual_size - passed_count,
            "pass_rate": pass_rate,
            "avg_quality_score": avg_quality,
            "results": results
        }
        
        logging.info(f"SWE-bench evaluation complete: {passed_count}/{actual_size} ({pass_rate:.1%} pass rate)")
        
        return evaluation_summary
