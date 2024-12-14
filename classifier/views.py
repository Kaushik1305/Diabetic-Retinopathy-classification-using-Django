import os
from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Create your views here.
from django.shortcuts import render
def home(request):
 return render(request,'home.html',{'name':'Raju'})

def predict_and_display(img_path):
    model = load_model('classifier/tensor_models/model_100_epochs.h5')
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = int(np.argmax(predictions))
    confidence_score = float(np.max(predictions))
    class_labels = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR','Severe']
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label, confidence_score

def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img = form.cleaned_data['image']
            img_path = f'classifier/uploads/{img.name}'
            
            # Save the uploaded image
            with open(img_path, 'wb+') as destination:
                for chunk in img.chunks():
                    destination.write(chunk)

            # Run the prediction
            predicted_class_index, confidence_score = predict_and_display(img_path)

            # Remove the uploaded image after prediction
            os.remove(img_path)

            return render(request, 'classifier/result.html', {
                'predicted_class_index': predicted_class_index,
                'confidence_score': confidence_score,
            })
    else:
        form = ImageUploadForm()

    return render(request, 'classifier/upload.html', {'form': form})