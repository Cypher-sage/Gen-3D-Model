# 3D Generation from Text or Image using SHAP-E

This project leverages OpenAI's [SHAP-E](https://github.com/openai/shap-e) model to convert text prompts or images into 3D models in `.obj` format. It includes background removal for cleaner image inputs and 3D mesh visualization using `trimesh`.

---

## 🔧 Steps to Run

1. **Clone SHAP-E Repository (if needed):**
   ```bash
   git clone https://github.com/openai/shap-e
   cd shap-e
   pip install -e .
   ```

2. **Install Required Libraries:**
   ```bash
   pip install torch rembg trimesh onnxruntime
   ```

3. **Run the Script:**
   ```bash
   python shap-e21.py
   ```

4. **Follow Prompts:**
   - For **text-to-3D**: input your prompt.
   - For **image-to-3D**: provide the image path and output directory.
   - At the end, specify the path to a `.obj` file to visualize the mesh.

---

## 📚 Libraries Used

- `torch` – runs models on CPU/GPU.
- `rembg` – removes backgrounds from input images.
- `trimesh` – displays generated 3D meshes.
- `onnxruntime` – supports SHAP-E model execution.
- `shap-e` – OpenAI's core 3D generation library.

---

![image](https://github.com/user-attachments/assets/c729e911-276c-4789-b52e-19388a5b50f0)
![image](https://github.com/user-attachments/assets/ba158c63-c1bd-44e4-bbbb-26da176b5003)
![image](https://github.com/user-attachments/assets/9af2b44d-6b3a-4355-b228-423db01138e8)
![3d image](https://github.com/user-attachments/assets/c26d530d-759c-4e4e-bbc5-c3460df3d309)

![image](https://github.com/user-attachments/assets/10aa1da8-e8ab-4c5e-8a57-e6caecc49b24)




## 🧠 Thought Process

This project was designed to offer a streamlined interface for converting both text and images into 3D models using OpenAI's SHAP-E. For **text-to-3D**, a simple prompt is passed to a pretrained text model (`text300M`) which generates a 3D representation through a diffusion-based sampling process. For **image-to-3D**, the image first goes through background removal to isolate the object of interest—this improves the model’s focus and output quality. The cleaned image is then processed by the image model (`image300M`) to produce the 3D mesh. In both cases, the final output is saved as a `.obj` file, and can be easily visualized using `trimesh`.
