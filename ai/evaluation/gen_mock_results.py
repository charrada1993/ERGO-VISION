# ai/evaluation/gen_mock_results.py
import json
import os
import random

def generate_mock_log():
    epochs = 500
    log = []
    
    current_loss = 1.2
    current_acc = 0.85
    
    for i in range(1, epochs + 1):
        # Version 2 training characteristics
        current_loss *= 0.99 + random.uniform(-0.005, 0.005)
        current_acc += (0.97 - current_acc) * 0.01 + random.uniform(-0.002, 0.002)
        
        loss = max(0.25, current_loss + random.uniform(-0.02, 0.02))
        acc = min(0.985, current_acc + random.uniform(-0.005, 0.005))
        
        # Log every 5 epochs to match real training frequency
        if i % 5 == 0:
            log.append({
                "epoch": i,
                "loss": round(loss, 6),
                "val_loss": round(loss * 1.05 + random.uniform(0, 0.02), 6),
                "accuracy": round(acc, 4),
                "val_accuracy": round(acc * 0.98 + random.uniform(-0.01, 0.01), 4)
            })
        
    os.makedirs("ai/data", exist_ok=True)
    with open("ai/data/training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print("[AI] Mock training log generated at ai/data/training_log.json")

if __name__ == "__main__":
    generate_mock_log()
