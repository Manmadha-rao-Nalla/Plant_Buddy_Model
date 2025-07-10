def predict_from_upload():
    try:
        root = Tk()
        root.withdraw() 

        file_path = filedialog.askopenfilename(
            title="Select a Leaf Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            print("❌ No file selected.")
            return

        img = Image.open(file_path).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 128, 128, 3)

        prediction = model.predict(img_array, verbose=0)
        class_index = np.argmax(prediction)
        confidence = float(np.max(prediction))

        disease_name = le.classes_[class_index]
        health_pct = round((1 - confidence) * 100, 2) if "healthy" not in disease_name else 100
        disease_pct = round(confidence * 100, 2) if "healthy" not in disease_name else 0

        print(f"\n✅ Prediction from Uploaded Image:")
        print(f"Disease Name: {disease_name}")
        print(f"Health %: {health_pct}%")
        print(f"Disease Severity %: {disease_pct}%")

    except Exception as e:
        print("❌ Error:", e)

predict_from_upload()
