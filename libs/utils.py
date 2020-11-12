def change_range(image, output_min, output_max, input_min=-1, input_max=1):
    output_image = ((image - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    if output_max > 1:
        output_image = output_image.int()
    return output_image
