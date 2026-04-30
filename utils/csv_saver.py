# utils/csv_saver.py
import csv
import os
from typing import Dict, List, Any, Optional

class CSVSaver:
    """Handles saving training results, including per-node CPU/Memory metrics."""
    def __init__(self, agent_name: str, save_dir: str = "results"):
        self.agent_name = agent_name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.filepath = os.path.join(save_dir, f"{agent_name}_results.csv")
        self.file_initialized = False

    def add_row(self, row_data: Dict[str, Any]):
        """Alias for save_row to maintain compatibility with training scripts."""
        self.save_row(row_data)
    
    def save_row(self, row_data: Dict[str, Any]):
        """Saves a single episode row. Dynamically handles per-node columns."""
        file_exists = os.path.isfile(self.filepath)
        
        # Open in append mode
        with open(self.filepath, 'a', newline='') as f:
            # We use DictWriter to handle dynamic per-node keys
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row_data)

class ResultsLoader:
    """Handles loading results for plotting with error-trapping for NoneType."""
    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir

    def load_all(self) -> Dict[str, List[Dict]]:
        results = {}
        if not os.path.exists(self.results_dir):
            return {}

        files = [f for f in os.listdir(self.results_dir) if f.endswith("_results.csv")]
        
        for f in files:
            agent_name = f.replace("_results.csv", "")
            filepath = os.path.join(self.results_dir, f)
            
            data = self._load_csv(filepath)
            
            # Fix for the 'NoneType' has no len() error
            if data: 
                results[agent_name] = data
                print(f"  ✓ Loaded {agent_name}: {len(data)} episodes")
            else:
                results[agent_name] = []
                print(f"  ⚠️ Skipping {agent_name}: File empty or corrupted.")
                
        return results

    def _load_csv(self, filepath: str) -> List[Dict]:
        data = []
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    converted_row = {}
                    for key, value in row.items():
                        try:
                            converted_row[key] = float(value)
                        except (ValueError, TypeError):
                            converted_row[key] = value
                    data.append(converted_row)
            return data
        except Exception as e:
            print(f"  [!] Error reading {filepath}: {e}")
            return [] # Always return a list to prevent TypeError