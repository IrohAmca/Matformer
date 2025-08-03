import signal
import sys

import ray
from ray import serve

from gemma import GemmaServeDeployment

print("Gemma Model Başlatılıyor...")
    
device = "cpu"
num_gpus = 1 if device == "cuda" else 0
print(f"Device: {device}")
    
ray.init(ignore_reinit_error=True)
serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
deployment = GemmaServeDeployment.options(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2}
).bind()
    
serve.run(deployment, route_prefix="/")
    
print("Hazır!")
print("http://localhost:8000")
print("Ctrl+C ile durdur")

def cleanup(sig, frame):
    print("\nDurduruluyor...")
    serve.shutdown()
    ray.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.CTRL_C_EVENT
