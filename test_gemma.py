import logging
import time
from gemma import GemmaE2B

import traceback
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_layer_skipping():
    print("=== Layer Skipping Testi ===")
    
    model = GemmaE2B("google/gemma-3n-e4b-it")
    
    prompt = "Python nedir?"
    layers_to_skip = [20, 21, 22, 23, 24]
    
    print(f"Prompt: {prompt}")
    print(f"Atlanan layer'lar: {layers_to_skip}")
    print("Generating with skipped layers...")
    
    start_time = time.time()
    
    try:
        result = model.generate_text(
            prompt=prompt,
            max_length=80,
            temperature=0.8,
            layers_to_skip=layers_to_skip
        )
        
        end_time = time.time()
        
        print(f"Sonuç: {result}")
        print(f"Süre: {end_time - start_time:.2f} saniye")
        
    except Exception as e:
        print(f"Hata: {str(e)}")
    
    print("-" * 50)


def main():
    print("Gemma Model Test Başlıyor...\n")
    
    try:        
        test_layer_skipping()
        
        print("Tüm testler tamamlandı!")
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()