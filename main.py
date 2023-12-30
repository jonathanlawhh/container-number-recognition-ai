import os
from typing import Tuple
import azure.ai.vision as sdk
import numpy as np
from dotenv import load_dotenv
import cv2
import json

load_dotenv()
assert os.environ["VISION_ENDPOINT"] != "", "Please ensure VISION ENDPOINT is setup in .env file."
assert os.environ["VISION_KEY"] != "", "Please ensure VISION KEY is setup in .env file."

service_options = sdk.VisionServiceOptions(os.environ["VISION_ENDPOINT"],
                                           os.environ["VISION_KEY"])


def downscale(ori_img: np.ndarray) -> np.ndarray:
    """
    If the image is >= 2000 pixels wide, downscale it proportionally
    :param ori_img: Input original image
    :return: Output resized image
    """
    h, w, c = ori_img.shape
    if w >= 1000:
        # If the value is 2000, take the down_factor as 2. If the value is 3000, take the factor as 3
        down_factor: int = (w // 1000 % 10) * 2
        return cv2.resize(ori_img, (w // down_factor, h // down_factor))

    return ori_img


def check_orientation_horizontal(bounding_box: list) -> bool:
    """
    Helper function to check if text is moving horizontally or vertically
    Simply, the difference between the x-axis compared with y-axis determines the orientation
    :param bounding_box: [x1, y1, x2, y2,...x6, y6]
    :return: True if it is horizontal, else False if it vertical
    """
    return bounding_box[2] - bounding_box[0] > bounding_box[6] - bounding_box[0]


def get_label_angle(bounding_box: list) -> int:
    """
    Compares the coordinates from 2 ends of a bounding box and find the difference
    :param bounding_box: [x1, y1, x2, y2,...x6, y6]
    :return: Pixel differences between angles
    """
    is_horizontal: bool = check_orientation_horizontal(bounding_box)
    if is_horizontal:
        return abs((bounding_box[3] - bounding_box[1]) + (bounding_box[5] - bounding_box[7])) // 2

    return abs((bounding_box[6] - bounding_box[0]) + (bounding_box[4] - bounding_box[2])) // 2


def within_buffer(co1_check_buff: int, co2: int, buffer: int = 50) -> bool:
    """
    Helper function to check if co2 is in between co1 with buffer
    :param co1_check_buff:
    :param co2:
    :param buffer:
    :return: True if co2 is within buffer, otherwise it is False
    """
    return co1_check_buff - buffer < co2 < co1_check_buff + buffer


def get_container_prefixes() -> Tuple:
    """
    Read the pre-defined container prefix and return in a Tuple
    :return: ("APHU", "EGHU"...)
    """
    with open("./container_prefix.txt", "r") as f:
        cp = tuple([line.split(",") for line in f if len(line) > 0][0])

    return cp if cp else tuple([])


def get_ctnr_color(ctnr_img: np.ndarray) -> np.ndarray:
    """
    Get the most dominant color from the image given
    :param ctnr_img: Input image
    :return: [B, G, R]
    """
    colors, count = np.unique(ctnr_img.reshape(-1, ctnr_img.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]


def detect_container_details(input_image_byte: bytes) -> json:
    """
    Takes in an image in byte array format, and run OCR on it
    :param input_image_byte: Image byte array []byte
    :return: json format of {"container_number": "ABCD1234567", "container_type": "45G1, "bounding_box": [x, y, x3, y3], "error": error_details.message}
    """

    # https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-analyze-image-40?pivots=programming-language-python
    # First convert the byte to an image source buffer
    image_source_buffer: sdk.ImageSourceBuffer = sdk.ImageSourceBuffer()
    image_source_buffer.image_writer.write(input_image_byte)
    vision_source = sdk.VisionSource(image_source_buffer=image_source_buffer)

    # Initialize the analysis options
    analysis_options: sdk.ImageAnalysisOptions = sdk.ImageAnalysisOptions()

    analysis_options.features = (
        sdk.ImageAnalysisFeature.TEXT
    )
    analysis_options.language = "en"
    image_analyzer = sdk.ImageAnalyzer(service_options, vision_source, analysis_options)

    # Analyze and get results
    result: sdk.ImageAnalysisResult = image_analyzer.analyze()

    # Define standard prefix, Tuple because it may run faster
    # carrier_prefix: list[str] = ["EGSU", "EMCU", "GAOU", "GLDU", "MRKU", "MRSU", "MSDU", "SEGU", "TGHU", "WEDU"]
    carrier_prefix: Tuple = get_container_prefixes()
    container_t_prefix: Tuple = ("G1", "R1", "U1", "P1", "T1")

    # The bounding block of detected container number
    # [x, y, x3, y3]
    bound_block = [0, 0, 0, 0]

    detected_container: str = ""
    detected_container_type: str = ""

    orientation_horizontal: bool = True
    last_xy1_cood: list[int, int] = []
    allowable_buffer: int = 50

    # Early exist if error, reduce nesting
    if result.reason != sdk.ImageAnalysisResultReason.ANALYZED:
        error_details = sdk.ImageAnalysisErrorDetails.from_result(result)
        return {"container_number": None, "container_type": None, "bounding_box": None, "error": error_details.message}

    if result.text is None:
        return {"container_number": None, "container_type": None, "bounding_box": None, "error": None}

    # Sort results by position
    # sorted_words = sorted(result.text.lines,
    #                       key=lambda item: ( round(item.bounding_polygon[1] / 100) * 100, round(item.bounding_polygon[0] / 100) * 100))

    for detected_text_line in result.text.lines:
        for word in detected_text_line.words:
            # As per shipping guidelines, container numbers will have 11 characters
            # https://www.evergreen-line.com/container/jsp/CNTR_ContainerMarkings.jsp
            if len(detected_container) >= 11 and detected_container_type != "":
                # Early exit if container number and container type is detected
                break

            clean_text: str = str(word.content).strip().replace(" ", "").upper()

            x1, y1, x2, y2, x3, y3, x4, y4 = word.bounding_polygon

            # Detect container prefix
            if detected_container == "" and any(prefix in clean_text for prefix in carrier_prefix):
                detected_container = clean_text
                orientation_horizontal = check_orientation_horizontal(word.bounding_polygon)
                last_xy1_cood = word.bounding_polygon[:2]

                bound_block[0] = x1
                bound_block[1] = y1
                bound_block[2] = x3
                bound_block[3] = y3

            # Detect container serial
            # Ensure the container prefix is populated first,
            # and the total character is >= 11 as per ISO standard
            if 11 > len(detected_container) >= 4:
                crit_met: bool = False

                # If horizontal container number
                if orientation_horizontal:
                    crit_met = x1 > last_xy1_cood[0] and within_buffer(last_xy1_cood[1], y1, allowable_buffer)

                # If vertical container number
                if not orientation_horizontal:
                    crit_met = y1 > last_xy1_cood[1] and within_buffer(last_xy1_cood[0], x1, allowable_buffer)

                if crit_met:
                    detected_container += clean_text
                    last_xy1_cood = word.bounding_polygon[:2]

                    bound_block[2] = x3 if x3 > bound_block[2] else bound_block[2]
                    bound_block[3] = y3 if y3 > bound_block[3] else bound_block[3]

            # Detect container type
            if detected_container_type == "":
                if any(t in clean_text for t in container_t_prefix):
                    detected_container_type = clean_text

            allowable_buffer = get_label_angle(word.bounding_polygon) * 3

    return {"container_number": detected_container, "container_type": detected_container_type,
            "bounding_box": [int(bb) for bb in bound_block],
            "error": None}


def http_request(request):
    from flask import jsonify
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """

    if 'image' not in request.files or request.files["image"].filename == '':
        return jsonify({"message": "No files found"}), 400

    file = request.files['image']
    if file and file.filename.rsplit('.', 1)[1].lower() not in ["jpg", "jpeg", "bmp", "png"]:
        return jsonify({"message": "Wrong file type"}), 415

    im = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    encoded_im = cv2.imencode('.JPG', im)[1].tobytes()

    return jsonify(detect_container_details(encoded_im)), 200


if __name__ == '__main__':
    image_dir = "./data"

    for filename in os.listdir(image_dir):
        f = os.path.join(image_dir, filename)

        # checking if it is a file
        if os.path.isfile(f) and f.endswith((".bmp", ".jpg", ".jpeg", ".png")):
            input_img = cv2.imread(f)
            a = cv2.imencode('.JPG', input_img)[1].tobytes()

            res: json = detect_container_details(a)
            assert res["error"] is None, res["error"]

            # Crop the detected bounding area from original image
            cropped_img: np.ndarray = input_img[max(0, res["bounding_box"][1] - 100): min(res["bounding_box"][3] + 100,
                                                                                          input_img.shape[1]),
                                      max(0, res["bounding_box"][0] - 100): min(res["bounding_box"][2] + 100,
                                                                                input_img.shape[0])]

            ctnr_color: np.ndarray = get_ctnr_color(cropped_img)
            res["container_color"] = ctnr_color.tolist()
            print(res)

            # Put the detected details top left of the cropped image
            text = f"{res['container_number']} - {res['container_type']}"
            cv2.putText(cropped_img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        color=(0, 255, 100), fontScale=1, thickness=2, lineType=cv2.LINE_AA)

            # Display
            # cropped_img = downscale(cropped_img)
            cv2.imshow("output", cropped_img)
            cv2.waitKey()
