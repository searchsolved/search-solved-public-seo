# eCommerce Image Centering Tool

A Streamlit application that automatically centres product images for consistent eCommerce displays.

## Features

- **Automatic detection** of the main subject in product images
- **Background removal** for products on white/light backgrounds
- **Consistent sizing** with customisable dimensions
- **WebP support**
- **Batch processing** for multiple images
- **Visual debugging** to see detection methods
- **Manual adjustments** for fine-tuning positioning

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Pillow

## Installation

```bash
git clone https://github.com/LeeFoot/ecommerce-image-centering.git
cd ecommerce-image-centering
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open your browser to http://localhost:8501

1. Upload your product images
2. Adjust settings in the sidebar if needed
3. Download processed images individually or in batch
4. Import into your eCommerce platform

## How It Works

The tool uses computer vision techniques to:

1. Detect the main subject using multiple thresholding methods
2. Remove backgrounds (optional)
3. Centre the product in the frame
4. Apply consistent padding
5. Export to your desired dimensions

## Example

| Original | Processed |
|----------|-----------|
| ![Original](https://via.placeholder.com/200x200.png?text=Original) | ![Processed](https://via.placeholder.com/200x200.png?text=Centered) |

## About the Author

**Lee Foot** - SEO and eCommerce consultant specializing in technical solutions for online retailers.

- 🌐 [Website](https://leefoot.co.uk)
- 🐦 [Twitter/X](https://x.com/LeeFootSEO/)
- ✉️ [Hire Me](mailto:hello@leefoot.co.uk)

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
