import numpy as np 
import matplotlib as plt 
from scipy.fft import fft2, ifft2, fftshift, ifftshift 
from skimage import data, color, img_as_float
import streamlit as st

# Konfigurasi halaman
st.set_page_config(page_title="Filter Frekuensi Fourier", layout="wide")

st.title("Aplikasi Filter Frekuensi Fourier 2D")
st.markdown("""
Aplikasi ini menunjukkan penerapan filter low-pass dan high-pass pada domain frekuensi 
menggunakan Transformasi Fourier 2D. Sesuaikan parameter menggunakan slider di sidebar.
""")

# Sidebar untuk parameter
st.sidebar.header("Filter Parameter")

# Parameter yang dapat diatur
radius = st.sidebar.slider("Radius Low-pass Filter", 
                          min_value=1, 
                          max_value=100, 
                          value=30,
                          help="Radius untuk filter low-pass (semakin besar semakin blur)")

# Load gambar
st.sidebar.header("Custom Image")
uploaded_file = st.sidebar.file_uploader("Upload file", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    # Load gambar custom (implementasi sederhana)
    import PIL.Image
    image = PIL.Image.open(uploaded_file)
    image = image.convert('L')  # Convert ke grayscale
    image = np.array(image) / 255.0  # Normalize ke [0,1]
else:
    image = color.rgb2gray(data.astronaut())
    image = img_as_float(image)

# Proses Fourier Transform
F = fft2(image) 
F_shifted = fftshift(F)
magnitude_spectrum = np.log(1 + np.abs(F_shifted))

# Membuat filter
rows, cols = image.shape 
crow, ccol = rows // 2, cols // 2 

# Low-pass mask
low_pass_mask = np.zeros_like(image) 
y, x = np.ogrid[:rows, :cols] 
mask_area = (x - ccol)**2 + (y - crow)**2 <= radius*radius 
low_pass_mask[mask_area] = 1 

# High-pass mask
high_pass_mask = 1 - low_pass_mask 

# Terapkan filter
low_pass_result = F_shifted * low_pass_mask 
high_pass_result = F_shifted * high_pass_mask 

# Kembali ke domain spasial
img_low = np.real(ifft2(ifftshift(low_pass_result))) 
img_high = np.real(ifft2(ifftshift(high_pass_result)))

# Normalisasi hasil
img_low = np.clip(img_low, 0, 1)
img_high = np.clip(img_high, 0, 1)

# Tampilkan hasil
st.header("Processing Result")

# Buat layout kolom
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Filter Mask")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    ax3.imshow(low_pass_mask, cmap='gray')
    ax3.set_title('Low-pass Mask')
    ax3.axis('off')
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Low-pass Filter Result")
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    ax4.imshow(img_low, cmap='gray')
    ax4.set_title('Hasil Low-pass (halus)')
    ax4.axis('off')
    st.pyplot(fig4)
    plt.close(fig4)

with col2:
    st.subheader("Frequency Spectrum")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.set_title('Spektrum Frekuensi (log)')
    ax2.axis('off')
    st.pyplot(fig2)
    plt.close(fig2)
    
    st.subheader("Filter Mask")
    fig5, ax5 = plt.subplots(figsize=(5, 4))
    ax5.imshow(high_pass_mask, cmap='gray')
    ax5.set_title('High-pass Mask')
    ax5.axis('off')
    st.pyplot(fig5)
    plt.close(fig5)

    st.subheader("High-pass Filter Result")
    fig6, ax6 = plt.subplots(figsize=(5, 4))
    ax6.imshow(img_high, cmap='gray')
    ax6.set_title('Hasil High-pass (tepi/detail)')
    ax6.axis('off')
    st.pyplot(fig6)
    plt.close(fig6)

# Penjelasan
with st.expander("Penjelasan Konsep"):
    st.markdown("""
    **Transformasi Fourier 2D** mengubah gambar dari domain spasial ke domain frekuensi.
    
    **Low-pass Filter:**
    - Menyimpan frekuensi rendah
    - Efek: Membuat gambar lebih halus/blur
    - Mengurangi noise dan detail halus
    
    **High-pass Filter:**
    - Menyimpan frekuensi tinggi  
    - Efek: Menonjolkan tepi dan detail
    - Berguna untuk deteksi tepi
    
    **Parameter:**
    - **Radius**: Mengontrol frekuensi yang digunakan 
    """)
