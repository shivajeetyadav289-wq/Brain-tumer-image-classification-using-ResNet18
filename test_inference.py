from src.inference import predict_image


image_path = "data/Testing/glioma/Te-gl_1.jpg"

predicted_class, confidence = predict_image(
    image_path
)

print("Predicted class:", predicted_class)
print(
    f"Confidence: {confidence * 100:.2f}%"
)