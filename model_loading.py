from tensorflow.keras.models import load_model # type: ignore

def load_trained_model(model_path='fps_optimization_model.h5'):
    
    model = load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model
