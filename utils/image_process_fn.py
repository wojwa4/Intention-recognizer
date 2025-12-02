import cv2

def resize_image_to_width(image, target_width: int = 400):

    original_height, original_width = image.shape[:2]
    scale_ratio = target_width / original_width
    new_height = int(original_height * scale_ratio)

    resized_image = cv2.resize(
        image,
        (target_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    return resized_image
