import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask
from PIL import Image

# --- Config ---
repo_url = "https://github.com/TheCrazyRocketScientist/Touch-Panel"
output_file = "qr_octocat_blue.png"
octocat_file = r"C:\Users\starm\Downloads\octocat.png"
qr_pixel_color = (0, 26, 79)
bg_color = (255, 255, 255)
dpi = 300
desired_cm_width = 9  # Target width in cm

# Step 1: Temporary QR to get module count
qr_temp = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    border=4,
    box_size=1  # placeholder
)
qr_temp.add_data(repo_url)
qr_temp.make(fit=True)
modules = qr_temp.modules_count  # Get size in modules (square)

# Step 2: Compute appropriate box size
inches = desired_cm_width / 2.54
pixel_width = int(inches * dpi)
box_size = pixel_width // modules

# Step 3: Final QR with correct box size
qr = qrcode.QRCode(
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    border=4,
    box_size=box_size
)
qr.add_data(repo_url)
qr.make(fit=True)

# --- Generate QR image ---
qr_img = qr.make_image(
    image_factory=StyledPilImage,
    module_drawer=RoundedModuleDrawer(),
    color_mask=SolidFillColorMask(
        front_color=qr_pixel_color,
        back_color=bg_color
    )
)

# --- Optional: Add Octocat ---
logo = Image.open(octocat_file).convert("RGBA")
qr_img = qr_img.convert("RGBA")
qr_width, qr_height = qr_img.size
logo_size = int(qr_width * 0.25)
logo = logo.resize((logo_size, logo_size), Image.LANCZOS)
pos = ((qr_width - logo_size) // 2, (qr_height - logo_size) // 2)
qr_img.paste(logo, pos, mask=logo)

# --- Save at 300 DPI ---
qr_img.save(output_file, dpi=(300, 300))
print(f"✅ Saved: {output_file} — {desired_cm_width}cm wide at 300 DPI")
