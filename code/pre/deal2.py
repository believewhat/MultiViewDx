import os
import tarfile
import shutil
import xml.etree.ElementTree as ET
import html
import json
import sys

def decode_html_entities(text):
    if text is not None:
        return html.unescape(text)
    return ''

def decode_unicode_escape(text):
    if text is not None:
        try:
            return bytes(text, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return text
    return ''

def extract_full_text(element):
    texts = []
    if element.text:
        texts.append(element.text)
    for sub_element in element:
        texts.append(extract_full_text(sub_element))
        if sub_element.tail:
            texts.append(sub_element.tail)
    return ''.join(texts)

def extract_images_and_captions(xml_root):
    images = []
    for fig in xml_root.findall('.//fig'):
        image_info = {}
        graphic = fig.find('.//graphic')
        if graphic is not None:
            image_info['image_filename'] = graphic.get('{http://www.w3.org/1999/xlink}href')
        
        label = fig.find('.//label')
        if label is not None:
            image_info['label'] = decode_unicode_escape(decode_html_entities(extract_full_text(label)))
        
        caption = fig.find('.//caption')
        if caption is not None:
            title = caption.find('.//title')
            if title is not None:
                image_info['title'] = decode_unicode_escape(decode_html_entities(extract_full_text(title)))
            
            caption_text = decode_unicode_escape(decode_html_entities(extract_full_text(caption)))
            image_info['caption'] = caption_text
        
        images.append(image_info)
    return images

def find_image_path(root, image_filename):
    possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp']
    for ext in possible_extensions:
        full_path = os.path.join(root, image_filename + ext)
        if os.path.exists(full_path):
            return full_path
    return None

def is_supported_image_format(file_path):
    supported_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.tif', '.tiff', '.bmp']
    return any(file_path.lower().endswith(ext) for ext in supported_extensions)

def process_tar_gz(file_path, extract_path, images_path, output_captions):
    captions_list = []
    with tarfile.open(file_path, 'r:gz') as tar:
        try:
            tar.extractall(path=extract_path)
        except:
            error = 1
        file_name = file_path.split('/')[-1].split('.')[0]
        for root, dirs, files in os.walk(os.path.join(extract_path, file_name)):
            for file in files:
                if file.endswith('.nxml'):
                    nxml_path = os.path.join(root, file)
                    tree = ET.parse(nxml_path)
                    xml_root = tree.getroot()
                    images_and_captions = extract_images_and_captions(xml_root)
                    
                    for image_info in images_and_captions:
                        image_filename = image_info.get('image_filename')
                        if image_filename:
                            src_image_path = find_image_path(root, image_filename)
                            if src_image_path and is_supported_image_format(src_image_path):
                                dst_image_dir = os.path.join(images_path, 'images')
                                os.makedirs(dst_image_dir, exist_ok=True)
                                dst_image_path = os.path.join(dst_image_dir, os.path.basename(src_image_path))
                                shutil.move(src_image_path, dst_image_path)
                                captions_list.extend(images_and_captions)
                                break
    shutil.rmtree(os.path.join(extract_path, file_name))

    with open(output_captions, 'w') as f:
        json.dump(captions_list, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python process_tar_gz.py <file_path> <extract_path> <output_captions>")
        sys.exit(1)

    file_path = sys.argv[1]
    extract_path = sys.argv[2]
    output_captions = sys.argv[3]

    process_tar_gz(file_path, extract_path, extract_path, output_captions)

