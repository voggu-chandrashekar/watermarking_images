import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog, Text, END
from PIL import Image, ImageTk
import os
import cv2
import pywt

# Load datasets
def preprocess_host_image(image):
    image = tf.image.resize(image, [256, 256])
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image

def preprocess_watermark_image(image):
    image = tf.image.resize(image, [64, 64])
    image = tf.image.rgb_to_grayscale(image)
    image = image / 255.0
    return image

def load_datasets(limit=100):
    # Load COCO dataset for host images with a limit on the number of images
    coco_ds = tfds.load('coco', split='train', shuffle_files=True)
    coco_ds = coco_ds.map(lambda sample: preprocess_host_image(sample['image'])).take(limit)

    # Create simple watermark images (e.g., QR codes, logos)
    # For simplicity, here we'll use a subset of COCO dataset as watermarks
    watermark_ds = tfds.load('coco', split='train', shuffle_files=True)
    watermark_ds = watermark_ds.map(lambda sample: preprocess_watermark_image(sample['image'])).take(limit)
    
    return coco_ds, watermark_ds

host_ds, watermark_ds = load_datasets(limit=100)

# Define the CNN model
def create_embedding_model():
    input_host = Input(shape=(256, 256, 1), name='host_image')
    input_watermark = Input(shape=(64, 64, 1), name='watermark_image')
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_host)
    watermark_upsampled = UpSampling2D((4, 4))(input_watermark)
    x = concatenate([x, watermark_upsampled])
    x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    model = Model(inputs=[input_host, input_watermark], outputs=x)
    model.compile(optimizer=Adam(), loss='mse')
    return model

embedding_model = create_embedding_model()

# Prepare the dataset for training
def prepare_dataset(host_ds, watermark_ds):
    dataset = tf.data.Dataset.zip((host_ds, watermark_ds))
    dataset = dataset.map(lambda host, watermark: ((host, watermark), host))
    dataset = dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

train_dataset = prepare_dataset(host_ds, watermark_ds)

# Train the model
embedding_model.fit(train_dataset, epochs=10)

# Save the model in TensorFlow SavedModel format
embedding_model.save('embedding_model.keras')

# Load the trained model
embedding_model = load_model('embedding_model.keras')

def cnn_embed_watermark(host_image, watermark_image):
    host_image = np.expand_dims(host_image, axis=(0, -1)) / 255.0
    watermark_image = np.expand_dims(watermark_image, axis=(0, -1)) / 255.0
    watermarked_image = embedding_model.predict([host_image, watermark_image])
    watermarked_image = np.squeeze(watermarked_image) * 255.0
    return watermarked_image.astype(np.uint8)

def calculate_metrics(original, watermarked):
    mse = np.mean((original - watermarked) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    ssim = 1 - mse / (np.var(original) + np.var(watermarked))
    return psnr, mse, ssim

class WatermarkingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermarking Images")

        # Buttons
        self.btn_upload_host = tk.Button(root, text="Upload Host Image", command=self.upload_host_image)
        self.btn_upload_host.pack()

        self.btn_upload_watermark = tk.Button(root, text="Upload Watermark Image", command=self.upload_watermark_image)
        self.btn_upload_watermark.pack()

        self.btn_cnn_watermarking = tk.Button(root, text="Run CNN Watermarking", command=self.run_cnn_watermarking)
        self.btn_cnn_watermarking.pack()

        self.btn_dwt_watermarking = tk.Button(root, text="Run DWT Watermarking", command=self.run_dwt_watermarking)
        self.btn_dwt_watermarking.pack()

        self.btn_extraction = tk.Button(root, text="Extraction", command=self.run_extraction)
        self.btn_extraction.pack()

        # Image containers
        self.cover_image_label = tk.Label(root)
        self.cover_image_label.pack()

        self.watermark_image_label = tk.Label(root)
        self.watermark_image_label.pack()

        self.watermarked_image_label = tk.Label(root)
        self.watermarked_image_label.pack()

        # Metrics
        self.metrics_label = tk.Label(root, text="")
        self.metrics_label.pack()

        self.text = Text(root, height=5, width=50)
        self.text.pack()

        self.host_image = None
        self.watermark_image = None
        self.watermarked_image = None

    def upload_host_image(self):
        file_path = filedialog.askopenfilename()
        self.host_image = Image.open(file_path).convert("L")
        self.host_image = self.host_image.resize((256, 256))  # Resize to the expected dimensions
        self.display_image(self.host_image, self.cover_image_label)

    def upload_watermark_image(self):
        file_path = filedialog.askopenfilename()
        self.watermark_image = Image.open(file_path).convert("L")
        self.watermark_image = self.watermark_image.resize((64, 64))  # Resize to the expected dimensions
        self.display_image(self.watermark_image, self.watermark_image_label)

    def run_cnn_watermarking(self):
        if self.host_image and self.watermark_image:
            self.watermarked_image = cnn_embed_watermark(np.array(self.host_image), np.array(self.watermark_image))
            self.display_image(Image.fromarray(self.watermarked_image), self.watermarked_image_label)
            psnr, mse, ssim = calculate_metrics(np.array(self.host_image), self.watermarked_image)
            self.metrics_label.config(text=f"CNN PSNR: {psnr} | CNN MSE: {mse} | CNN SSIM: {ssim}")

    def run_dwt_watermarking(self):
        if self.host_image and self.watermark_image:
            self.host_image_path = filedialog.askopenfilename(title="Select Host Image")
            self.watermark_image_path = filedialog.askopenfilename(title="Select Watermark Image")
            cover_image = cv2.imread(self.host_image_path, 0)
            watermark_image = cv2.imread(self.watermark_image_path, 0)

            cover_image = cv2.resize(cover_image, (300, 300))
            cv2.imshow('Cover Image', cv2.resize(cover_image, (400, 400)))
            watermark_image = cv2.resize(watermark_image, (150, 150))
            cv2.imshow('Watermark Image', cv2.resize(watermark_image, (400, 400)))

            cover_image = np.float32(cover_image)
            cover_image /= 255
            coeffC = pywt.dwt2(cover_image, 'haar')
            cA, (cH, cV, cD) = coeffC
            watermark_image = np.float32(watermark_image)
            watermark_image /= 255

            # Embedding
            coeffW = (0.4 * cA + 0.1 * watermark_image, (cH, cV, cD))
            watermarked_image = pywt.idwt2(coeffW, 'haar')
            psnr, mse = self.PSNR(cover_image, watermarked_image)
            ssim = self.imageSSIM(cover_image, watermarked_image)
            self.text.insert(END, "DWT PSNR : " + str(psnr) + "\n")
            self.text.insert(END, "DWT MSE  : " + str(mse) + "\n")
            self.text.insert(END, "DWT SSIM : " + str(ssim) + "\n\n")
            self.text.update_idletasks()
            name = os.path.basename(self.host_image_path)
            cv2.imwrite("OutputImages/" + name, watermarked_image * 255)
            np.save("model/" + name, watermarked_image)
            np.save("model/CA_" + name, cA)
            cv2.imshow('Watermarked Image', cv2.resize(watermarked_image, (400, 400)))
            cv2.waitKey(0)

    def run_extraction(self):
        wm = filedialog.askopenfilename(initialdir="OutputImages")
        wm = os.path.basename(wm)
        img = np.load("model/" + wm + ".npy")
        cA = np.load("model/CA_" + wm + ".npy")
        coeffWM = pywt.dwt2(img, 'haar')
        hA, (hH, hV, hD) = coeffWM

        extracted = (hA - 0.4 * cA) / 0.1
        extracted *= 255
        extracted = np.uint8(extracted)
        extracted = cv2.resize(extracted, (400, 400))
        cv2.imshow('Extracted Image', extracted)
        cv2.waitKey(0)

    def display_image(self, image, label):
        imgtk = ImageTk.PhotoImage(image=image)
        label.config(image=imgtk)
        label.image = imgtk

    def PSNR(self, original, watermarked):
        mse = np.mean((original - watermarked) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        return psnr, mse

    def imageSSIM(self, original, watermarked):
        ssim = 1 - np.mean((original - watermarked) ** 2) / (np.var(original) + np.var(watermarked))
        return ssim

if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkingApp(root)
    root.mainloop()

