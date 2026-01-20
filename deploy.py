import signal
import sys

import ray
from ray import serve

from gemma import GemmaServeDeployment

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.info("Gemma Model Başlatılıyor...")
    
device = "cuda"
num_gpus = 1 if device == "cuda" else 0
logging.info(f"Device: {device}")
    
ray.init(ignore_reinit_error=True)
serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    
deployment = GemmaServeDeployment.options(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2}
).bind()
    
serve.run(deployment, route_prefix="/")
    
logging.info("Hazır!")
logging.info("http://localhost:8000")
logging.info("Ctrl+C ile durdur")

def cleanup(sig, frame):
    logging.info("\nDurduruluyor...")
    serve.shutdown()
    ray.shutdown()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.CTRL_C_EVENT
