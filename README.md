# **Compresor y Descompresor JPEG**

Este proyecto implementa un algoritmo de compresión y descompresión de imágenes utilizando el formato **JPEG**, uno de los estándares más populares para la compresión de imágenes digitales. 

---

## **¿Qué es JPEG?**

JPEG (Joint Photographic Experts Group) es un estándar para la compresión con pérdida de imágenes. Su principal ventaja es reducir significativamente el tamaño de los archivos manteniendo una calidad visual aceptable para el ojo humano. Es ideal para fotografías y gráficos complejos.

### **Pasos principales del algoritmo JPEG**

1. **Conversión de color (RGB a YCbCr):**
   - La imagen se transforma del espacio de color RGB al espacio YCbCr.  
   - **Y (Luminancia):** Información de brillo.  
   - **Cb, Cr (Crominancia):** Información de color.

2. **Submuestreo de crominancia:**
   - Los componentes de color (Cb y Cr) se submuestrean (generalmente 4:2:0) para reducir su resolución, ya que el ojo humano es menos sensible a los cambios de color que a los cambios de brillo.

3. **Aplicación de padding:**
   - Para garantizar que las dimensiones de la imagen sean múltiplos de 8 (requerido por las transformaciones posteriores), se agrega relleno (padding) si es necesario.

4. **Transformada Discreta del Coseno (DCT):**
   - Convierte cada bloque de 8x8 desde el dominio espacial al dominio de frecuencias, separando información importante (frecuencias bajas) de información menos perceptible (frecuencias altas).

5. **Cuantización:**
   - Las frecuencias altas se reducen (o eliminan) dividiendo los coeficientes del bloque por una matriz de cuantización. La calidad de compresión se controla ajustando esta matriz.

6. **Codificación Zigzag y RLE:**
   - Los coeficientes se recorren en orden zigzag para agrupar ceros consecutivos, que luego se comprimen mediante **Run-Length Encoding (RLE)**.

7. **Codificación Huffman:**
   - Se utiliza para comprimir aún más los datos mediante una codificación basada en frecuencias.

8. **Almacenamiento:**
   - Se guarda la información comprimida junto con metadatos necesarios para descomprimir la imagen.

9. **Descompresión:**
   - Es el proceso inverso de los pasos anteriores, reconstruyendo la imagen original lo más cerca posible de la original.

---

## **Estructura del proyecto**

```
📁 JPEGCompressor
├── img/
├── src/
│   ├── bgr_ycrcb_conversion.py
│   ├── functions.py
│   ├── discrete_cosine_transform.py
│   ├── quantization.py
│   ├── run_length_encoding.py
│   ├── huffman_encoding.py
│   ├── metadata.py
├── app.py
├── main.py
├── README.md
├── .gitignore
└── requirements.txt
```

---

## **Cómo usar la aplicación**

...

---
