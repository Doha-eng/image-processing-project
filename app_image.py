import streamlit as st
import cv2
import numpy as np
from PIL import Image
from image_processor import ImageProcessor
import io

st.set_page_config(page_title="Image Processing Project", layout="wide")

st.title("🖼️ Image Processing Project")

# Initialize session state for processed image
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'operation_name' not in st.session_state:
    st.session_state.operation_name = ""

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("Image was uploaded successfully")
    
    # Convert the file to an opencv image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    st.sidebar.header("Select Operation")
    category = st.sidebar.selectbox("Category", [
        "1. Point Operation",
        "2. Color Image Operation",
        "3. Image Histogram",
        "4. Neighborhood Processing",
        "5. Image Restoration",
        "6. Image Segmentation",
        "7. Edge Detection",
        "8. Mathematical Morphology"
    ])

    
    if st.sidebar.button("🔄 Reset Image"):
        st.session_state.processed_image = None
        st.session_state.operation_name = ""
        st.rerun()

    
    if category == "1. Point Operation":
        op = st.sidebar.radio("Operation", ["Addition", "Subtraction", "Division", "Complement"])
        val = st.sidebar.slider("Value", 0, 255, 50)
        if st.sidebar.button("Apply Filter"):
            st.session_state.operation_name = op
            if op == "Addition":
                st.session_state.processed_image = ImageProcessor.addition(original_image, val)
            elif op == "Subtraction":
                st.session_state.processed_image = ImageProcessor.subtraction(original_image, val)
            elif op == "Division":
                st.session_state.processed_image = ImageProcessor.division(original_image, val)
            elif op == "Complement":
                st.session_state.processed_image = ImageProcessor.complement(original_image)

    elif category == "2. Color Image Operation":
        op = st.sidebar.radio("Operation", ["Change Lighting Color (Red)", "Swap R to G", "Eliminate Red"])
        if op == "Change Lighting Color (Red)":
            val = st.sidebar.slider("Red Value", -255, 255, 50)
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                st.session_state.processed_image = ImageProcessor.change_lighting_color(original_image, val)
        elif op == "Swap R to G":
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                st.session_state.processed_image = ImageProcessor.swap_channels_r_g(original_image)
        elif op == "Eliminate Red":
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                st.session_state.processed_image = ImageProcessor.eliminate_red(original_image)

    elif category == "3. Image Histogram":
        op = st.sidebar.radio("Operation", ["Histogram Stretching", "Histogram Equalization"])
        if st.sidebar.button("Apply Filter"):
            st.session_state.operation_name = op
            if op == "Histogram Stretching":
                st.session_state.processed_image = ImageProcessor.histogram_stretching(original_image)
            elif op == "Histogram Equalization":
                st.session_state.processed_image = ImageProcessor.histogram_equalization(original_image)

    elif category == "4. Neighborhood Processing":
        filter_type = st.sidebar.selectbox("Filter Type", ["Linear", "Non-Linear"])
        if filter_type == "Linear":
            op = st.sidebar.radio("Operation", ["Average Filter", "Laplacian Filter"])
            if op == "Average Filter":
                size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
                if st.sidebar.button("Apply Filter"):
                    st.session_state.operation_name = op
                    st.session_state.processed_image = ImageProcessor.average_filter(original_image, size)
            elif op == "Laplacian Filter":
                if st.sidebar.button("Apply Filter"):
                    st.session_state.operation_name = op
                    st.session_state.processed_image = ImageProcessor.laplacian_filter(original_image)
        else:
            op = st.sidebar.radio("Operation", ["Maximum", "Minimum", "Median", "Most Frequent (Mode)"])
            size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                if op == "Maximum":
                    st.session_state.processed_image = ImageProcessor.max_filter(original_image, size)
                elif op == "Minimum":
                    st.session_state.processed_image = ImageProcessor.min_filter(original_image, size)
                elif op == "Median":
                    st.session_state.processed_image = ImageProcessor.median_filter(original_image, size)
                elif op == "Most Frequent (Mode)":
                    st.session_state.processed_image = ImageProcessor.mode_filter(original_image, size)

    elif category == "5. Image Restoration":
        noise_type = st.sidebar.selectbox("Noise Type", ["Salt and Pepper", "Gaussian"])
        if noise_type == "Salt and Pepper":
            noisy_img = ImageProcessor.add_salt_and_pepper(original_image)
            st.subheader("Noisy Image (Salt & Pepper)")
            st.image(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            op = st.sidebar.radio("Restoration Method", ["Average Filter", "Median Filter", "Outlier Method"])
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                if op == "Average Filter":
                    st.session_state.processed_image = ImageProcessor.average_filter(noisy_img)
                elif op == "Median Filter":
                    st.session_state.processed_image = ImageProcessor.median_filter(noisy_img)
                elif op == "Outlier Method":
                    st.session_state.processed_image = ImageProcessor.outlier_method(noisy_img)
        else:
            noisy_img = ImageProcessor.add_gaussian_noise(original_image)
            st.subheader("Noisy Image (Gaussian)")
            st.image(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            op = st.sidebar.radio("Restoration Method", ["Image Averaging (Simulated)", "Average Filter"])
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                if op == "Image Averaging (Simulated)":
                    avg_img = noisy_img.astype(np.float32)
                    for _ in range(5):
                        avg_img += ImageProcessor.add_gaussian_noise(original_image).astype(np.float32)
                    st.session_state.processed_image = (avg_img / 6).astype(np.uint8)
                elif op == "Average Filter":
                    st.session_state.processed_image = ImageProcessor.average_filter(noisy_img)

    elif category == "6. Image Segmentation":
        op = st.sidebar.radio("Thresholding Type", ["Basic Global", "Automatic (Otsu)", "Adaptive"])
        if op == "Basic Global":
            thresh = st.sidebar.slider("Threshold Value", 0, 255, 127)
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                st.session_state.processed_image = ImageProcessor.thresholding_basic(original_image, thresh)
        elif op == "Automatic (Otsu)":
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                st.session_state.processed_image = ImageProcessor.thresholding_automatic(original_image)
        elif op == "Adaptive":
            if st.sidebar.button("Apply Filter"):
                st.session_state.operation_name = op
                st.session_state.processed_image = ImageProcessor.thresholding_adaptive(original_image)

    elif category == "7. Edge Detection":
        if st.sidebar.button("Apply Filter"):
            st.session_state.operation_name = "Sobel Detector"
            st.session_state.processed_image = ImageProcessor.sobel_detector(original_image)

    elif category == "8. Mathematical Morphology":
        op = st.sidebar.radio("Operation", ["Dilation", "Erosion", "Opening", "Internal Boundary", "External Boundary", "Morphological Gradient"])
        size = st.sidebar.slider("Kernel Size", 3, 15, 3, step=2)
        if st.sidebar.button("Apply Filter"):
            st.session_state.operation_name = op
            if op == "Dilation":
                st.session_state.processed_image = ImageProcessor.dilation(original_image, size)
            elif op == "Erosion":
                st.session_state.processed_image = ImageProcessor.erosion(original_image, size)
            elif op == "Opening":
                st.session_state.processed_image = ImageProcessor.opening(original_image, size)
            elif op == "Internal Boundary":
                st.session_state.processed_image = ImageProcessor.internal_boundary(original_image)
            elif op == "External Boundary":
                st.session_state.processed_image = ImageProcessor.external_boundary(original_image)
            elif op == "Morphological Gradient":
                st.session_state.processed_image = ImageProcessor.morphological_gradient(original_image)

    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(original_image_rgb, use_container_width=True)
    
    with col2:
        if st.session_state.processed_image is not None:
            st.subheader(f"Processed: {st.session_state.operation_name}")
            
            
            if len(st.session_state.processed_image.shape) == 2:
                st.image(st.session_state.processed_image, use_container_width=True)
                display_img = st.session_state.processed_image
            else:
                display_img_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
                st.image(display_img_rgb, use_container_width=True)
                display_img = display_img_rgb
            
            
            result_pil = Image.fromarray(display_img)
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="💾 Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )
        else:
            st.subheader("Result")
            st.info("Select a filter and click 'Apply Filter' to see the result.")
else:
    st.info("Please upload an image to start.")
