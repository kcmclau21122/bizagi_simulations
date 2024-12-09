from PIL import Image

def png_to_high_res_tiff(png_path, tiff_path, dpi=(1200, 1200)):
    """
    Convert a PNG file to a high-resolution TIFF file.
    
    :param png_path: Path to the PNG file.
    :param tiff_path: Path to save the output TIFF file.
    :param dpi: Tuple specifying the DPI (dots per inch) for the TIFF.
    """
    # Open the PNG file
    with Image.open(png_path) as img:
        # Save the image as a high-resolution TIFF
        img.save(tiff_path, format="TIFF", dpi=dpi, compression="tiff_lzw")
    print(f"Converted {png_path} to {tiff_path} with DPI {dpi}")

# Example usage
png_path = "C:/Test Data/Bizagi/5.5.13 Real Property-Monthly_Reviews.png"  # Path to your PNG file
tiff_path = "C:/Test Data/Bizagi/5.5.13 Real Property-Monthly_Reviews.tiff"  # Desired path for the high-resolution TIFF file
png_to_high_res_tiff(png_path, tiff_path, dpi=(600, 600))  # Set DPI to 600x600
