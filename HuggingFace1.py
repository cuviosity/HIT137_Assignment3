import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from transformers import pipeline
import threading

# Base class demonstrating inheritance and encapsulation
class AIModel:
    """Base class for AI models - demonstrates encapsulation"""
    def __init__(self, model_name, task):
        self._model_name = model_name  # Private attribute (encapsulation)
        self._task = task
        self._pipeline = None
    
    def load_model(self):
        """Load the model pipeline"""
        print(f"Loading {self._model_name}...")
        self._pipeline = pipeline(self._task, model=self._model_name)
        print("Model loaded successfully!")
    
    def predict(self, image_path):
        """Method to be overridden - demonstrates polymorphism"""
        raise NotImplementedError("Subclasses must implement predict()")
    
    def get_model_info(self):
        """Returns model information"""
        return f"Model: {self._model_name}\nTask: {self._task}"


# Child class 1 - demonstrates inheritance
class ImageClassifierModel1(AIModel):
    """Google ViT model for image classification"""
    def __init__(self):
        super().__init__("google/vit-base-patch16-224", "image-classification")
    
    def predict(self, image_path):
        """Override predict method - demonstrates method overriding"""
        image = Image.open(image_path)
        results = self._pipeline(image)
        return results


# Child class 2 - demonstrates inheritance  
class ImageClassifierModel2(AIModel):
    """Microsoft ResNet model for image classification"""
    def __init__(self):
        super().__init__("microsoft/resnet-50", "image-classification")
    
    def predict(self, image_path):
        """Override predict method - demonstrates method overriding"""
        image = Image.open(image_path)
        results = self._pipeline(image)
        return results


# Decorator pattern - adds preprocessing/postprocessing
class ModelDecorator:
    """Decorator to add extra functionality - demonstrates decorator pattern"""
    def __init__(self, model):
        self._model = model
    
    def predict(self, image_path):
        """Wraps prediction with additional processing"""
        print("Preprocessing image...")
        result = self._model.predict(image_path)
        print("Postprocessing results...")
        return result
    
    def get_model_info(self):
        return self._model.get_model_info()


# Main GUI Application
class ImageClassifierGUI:
    """Main application demonstrating multiple OOP concepts"""
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Classifier - OOP Demo")
        self.root.geometry("900x700")
        
        # Initialize models (lazy loading)
        self.models = {
            "Google ViT": None,
            "Microsoft ResNet": None
        }
        self.current_image_path = None
        self.current_photo = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title = tk.Label(self.root, text="Image Classification with Hugging Face", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Model selection frame
        model_frame = tk.LabelFrame(self.root, text="Select AI Model", padx=10, pady=10)
        model_frame.pack(padx=20, pady=10, fill="x")
        
        self.model_var = tk.StringVar(value="Google ViT")
        model_options = ["Google ViT", "Microsoft ResNet"]
        
        for option in model_options:
            rb = tk.Radiobutton(model_frame, text=option, variable=self.model_var, 
                               value=option, font=("Arial", 11))
            rb.pack(anchor="w")
        
        # Image upload section
        upload_frame = tk.Frame(self.root)
        upload_frame.pack(pady=10)
        
        upload_btn = tk.Button(upload_frame, text="📁 Upload Image", 
                              command=self.upload_image, font=("Arial", 12),
                              bg="#4CAF50", fg="white", padx=20, pady=10)
        upload_btn.pack()
        
        # Image display
        self.image_label = tk.Label(self.root, text="No image uploaded", 
                                   bg="lightgray", width=50, height=15)
        self.image_label.pack(pady=10)
        
        # Classify button
        classify_btn = tk.Button(self.root, text="🔍 Classify Image", 
                                command=self.classify_image, font=("Arial", 12),
                                bg="#2196F3", fg="white", padx=20, pady=10)
        classify_btn.pack(pady=10)
        
        # Results display
        results_frame = tk.LabelFrame(self.root, text="Classification Results", 
                                     padx=10, pady=10)
        results_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        self.results_text = tk.Text(results_frame, height=6, font=("Arial", 11))
        self.results_text.pack(fill="both", expand=True)
        
        # OOP Explanation section
        self.create_info_section()
    
    def create_info_section(self):
        """Create information section with Model Info and OOP Explanation side by side"""
        info_frame = tk.LabelFrame(self.root, text="Model Information & OOP Explanation", 
                                   padx=10, pady=10)
        info_frame.pack(padx=20, pady=10, fill="both", expand=True)
        
        # Create two columns
        left_frame = tk.Frame(info_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        right_frame = tk.Frame(info_frame)
        right_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Left side - Selected Model Info
        model_label = tk.Label(left_frame, text="Selected Model Info:", 
                              font=("Arial", 11, "bold"))
        model_label.pack(anchor="w", pady=(0, 5))
        
        model_text = tk.Text(left_frame, wrap="word", font=("Arial", 9), 
                            height=15, bg="#f0f0f0")
        model_text.pack(fill="both", expand=True)
        
        model_info = """• Model Name:
  - Google ViT (Vision Transformer)
  - Microsoft ResNet-50

• Category:
  - Image Classification

• Short Description:
  Both models are pre-trained on ImageNet 
  with 1000 object categories. They analyze 
  images and predict what objects are 
  present with confidence scores.
  
  Google ViT uses modern transformer 
  architecture for high accuracy.
  
  Microsoft ResNet-50 uses residual 
  connections for efficient and fast 
  classification.
  
• Input: Image files (JPG, PNG, BMP)
• Output: Top 5 predictions with confidence
"""
        model_text.insert("1.0", model_info)
        model_text.config(state="disabled")
        
        # Right side - OOP Concepts Explanation
        oop_label = tk.Label(right_frame, text="OOP Concepts Explanation:", 
                            font=("Arial", 11, "bold"))
        oop_label.pack(anchor="w", pady=(0, 5))
        
        oop_text = tk.Text(right_frame, wrap="word", font=("Arial", 9), 
                          height=15, bg="#f0f0f0")
        oop_text.pack(fill="both", expand=True)
        
        oop_explanation = """• Where Multiple Inheritance is used:
  - ImageClassifierModel1 and 
    ImageClassifierModel2 both inherit from 
    the AIModel base class (lines 10-23)
  - They will assume load_model() and 
    get_model_info() methods

• Why Encapsulation was applied:
  - Private attributes (_model_name, 
    _pipeline) hide internal data (line 12-13)
  - The data can only be accessed through public 
    methods, preventing immediate manipulation
  - This shields model stability and makes 
    code maintainable

• How Polymorphism and Method Overriding 
  are shown:
  - predict() method is defined in parent 
    AIModel class (line 21-23)
  - Each sub class overrides predict() 
    with self implementation (lines 31, 41)
  - Same method name, different behaviors - 
    this is polymorphism in action

• Where Multiple Decorators are applied:
  - ModelDecorator class wraps any model 
    (line 46-59)
  - Adds preprocessing/postprocessing without 
    changing original model classes
  - Applied in run_classification() method 
    (line 177) when wrapping models
"""
        oop_text.insert("1.0", oop_explanation)
        oop_text.config(state="disabled")
    
    def upload_image(self):
        """Handle image upload"""
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
    
    def display_image(self, image_path):
        """Display uploaded image in GUI"""
        image = Image.open(image_path)
        # Resize to fit display
        image.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo  # Keep reference
    
    def classify_image(self):
        """Classify the uploaded image using selected model"""
        if not self.current_image_path:
            self.results_text.delete("1.0", tk.END)
            self.results_text.insert("1.0", "Please upload an image first!")
            return
        
        selected_model = self.model_var.get()
        
        # Show loading message
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", f"Loading {selected_model} model and classifying...\nThis may take a moment...")
        self.root.update()
        
        # Run in thread to prevent GUI freezing
        thread = threading.Thread(target=self.run_classification, args=(selected_model,))
        thread.start()
    
    def run_classification(self, selected_model):
        """Run classification in separate thread"""
        try:
            # Load model if not already loaded (lazy loading)
            if self.models[selected_model] is None:
                if selected_model == "Google ViT":
                    model = ImageClassifierModel1()
                else:
                    model = ImageClassifierModel2()
                
                model.load_model()
                # Wrap with decorator
                self.models[selected_model] = ModelDecorator(model)
            
            # Get prediction
            model = self.models[selected_model]
            results = model.predict(self.current_image_path)
            
            # Format results
            output = f"Results from {selected_model}:\n\n"
            for i, result in enumerate(results[:5], 1):  # Top 5 results
                label = result['label']
                score = result['score'] * 100
                output += f"{i}. {label}: {score:.2f}%\n"
            
            # Update GUI (must be done in main thread)
            self.root.after(0, self.update_results, output)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, self.update_results, error_msg)
    
    def update_results(self, text):
        """Update results text (called from main thread)"""
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", text)


# Main execution
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()
