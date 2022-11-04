import xml.etree.ElementTree as ET

tree = ET.parse('scripts\Czech_000086.xml')
root = tree.getroot()


for obj in root.iter('object'):
    label = obj[0].text
    bounding_box = obj[1]
    print(f"{label}: {bounding_box[0].text, bounding_box[1].text, bounding_box[2].text, bounding_box[3].text}")
    print(int(bounding_box[0].text) + int(bounding_box[1].text))
    print(len(bounding_box))
    #print(obj.attrib[0])

#models = file.getElementsByTagName('model')