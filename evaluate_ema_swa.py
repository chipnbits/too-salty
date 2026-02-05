
import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch


from evaluate import evaluate_model 
from salty.datasets import get_cifar100_loaders, get_cifar100c_loaders_by_corruption
from salty.models import get_resnet50_model
from salty.permute_model import permute_models
from salty.similarity_metrics_from_logits import cka_similarity, logit_mse_kl
from salty.similarity_metrics_from_models import cosine_similarity, l2_distance
from salty.utils import load_checkpoint, soup_models

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
EMA_SWA_DIR = os.path.join(MODEL_DIR, "cifar100-resnet50/ema_swa_experiments")

class EMA_SWA_Loader:
    """
    Load and hande files from a single EMA/SWA directory containing:
    - Best baseline model (validation accuracy)
    - Best EMA model (validation accuracy)
    - Final EMA model (last epoch)
    - Final SWA model (last epoch)
    """
    
    def __init__(self, ema_swa_dir: Path, device):
        self.ema_swa_dir = ema_swa_dir
        self.device = device     
        
        parts = self.ema_swa_dir.name.split("_")
        assert parts[0] == "run", f"Directory name should start with 'run_', got {self.ema_swa_dir.name}"
        assert parts[2] == "seed", f"Directory name should contain '_seed_', got {self.ema_swa_dir.name}"
        self.run_id = parts[1].replace("run", "")
        self.seed = parts[3].replace("seed", "")
        
            
    def load_model(self, checkpoint_path):
        """
        Loads weights from a checkpoint dictionary into the model.
        Handles 'model_state' key and potential DataParallel wrapping (module. prefix).
        """
        model = get_resnet50_model(num_classes=100).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        return model
    
    @property
    def baseline_best(self):
        path = self.ema_swa_dir / "baseline_best.pt"
        return self.load_model(path)
    
    # @property
    # def ema_best(self):
    #     path = self.ema_swa_dir / "ema_best.pt"
    #     return self.load_model(path)
    
    @property
    def ema_final(self):
        path = self.ema_swa_dir / "ema_final.pt"
        return self.load_model(path)
    
    @property
    def swa_final(self):
        path = self.ema_swa_dir / "swa_final.pt"
        return self.load_model(path)
    
    def __iter__(self):
        """
        Iterates through the 3 key models: Baseline, EMA, SWA.
        Yields: (model_type_string, model_object)
        """
        # We load them one by one to save GPU memory during iteration
        yield "Baseline", self.baseline_best
        yield "EMA", self.ema_final
        yield "SWA", self.swa_final

def run_ema_swa_evaluation(
    root_dir: str, 
    clean_loader, 
    corrupted_loaders, 
    device,
    output_csv: str = "ema_swa_robustness_results.csv"
):
    """
    Iterates over all EMA/SWA run folders, evaluates the 3 models (Baseline, EMA, SWA)
    on clean and corrupted data, and aggregates results.
    """
    root_path = Path(root_dir)
    results = []
    
    # Check if results file already exists to resume or append
    if os.path.exists(output_csv):
        print(f"Resuming/Appending to {output_csv}...")
        existing_df = pd.read_csv(output_csv)
        # Create a set of processed (run_id, model_type) to skip duplicates
        processed_keys = set(zip(existing_df["run_id"].astype(str), existing_df["model_type"]))
    else:
        print(f"Creating new results file {output_csv}...")
        processed_keys = set()

    # Find all run directories
    run_dirs = sorted([d for d in root_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    print(f"Found {len(run_dirs)} run directories in {root_dir}")

    for run_dir in tqdm(run_dirs, desc="Processing Runs"):
        model_loader = EMA_SWA_Loader(run_dir, device)
        
        if model_loader.run_id in ["12", "13"]:  # skip these runs in progress
            print(f"Skipping run_id {model_loader.run_id} due to incomplete run.")
            continue

        # Iterate through Baseline, EMA, SWA
        for model_type, model in model_loader:
            
            # Skip if already processed
            if (model_loader.run_id, model_type) in processed_keys:
                continue
            
            clean_acc, clean_loss = evaluate_model(model, clean_loader, device)
            
            corruption_accs = {}
            for corr_name, loader in corrupted_loaders.items():
                corr_acc, corr_loss = evaluate_model(model, loader, device)
                corruption_accs[corr_name] = (corr_acc, corr_loss)
            
            # Aggregate corrupted accuracies (mean over corruptions)
            corr_acc = sum(acc for acc, _ in corruption_accs.values()) / len(corruption_accs)
            corr_loss = sum(loss for _, loss in corruption_accs.values()) / len(corruption_accs)
            
            result_row = {
                "run_id": model_loader.run_id,
                "seed": model_loader.seed,
                "model_type": model_type,
                "clean_acc": clean_acc,
                "clean_loss": clean_loss,
                "corrupted_acc": corr_acc,
                "corrupted_loss": corr_loss
            }
            results.append(result_row)
            
            # Save incrementally
            df_new = pd.DataFrame([result_row])
            header = not os.path.exists(output_csv)
            df_new.to_csv(output_csv, mode="a", header=header, index=False)
            
            # Memory Cleanup
            del model
            torch.cuda.empty_cache()

    # Final Save to Parquet for efficiency
    if os.path.exists(output_csv):
        df_final = pd.read_csv(output_csv)
        parquet_path = output_csv.replace(".csv", ".parquet")
        df_final.to_parquet(parquet_path)
        print(f"\n✅ Evaluation complete. Results saved to:\n  - {output_csv}\n  - {parquet_path}")

if __name__ == "__main__":
    # walk ema_swa directory for models
    path_obj = Path(EMA_SWA_DIR)
    batch_size = 128
    severity = 3

    # for each subdirectory, load models and evaluate
    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda:0"
    _,_, test_loader = get_cifar100_loaders(batch_size=128)
    corruption_loaders = get_cifar100c_loaders_by_corruption(
                batch_size=batch_size,
                severity=severity,
            )
    
    run_ema_swa_evaluation(
        root_dir=EMA_SWA_DIR,
        clean_loader=test_loader,
        corrupted_loaders=corruption_loaders,
        device=device,
        output_csv="ema_swa_robustness_results.csv"
    )
    
