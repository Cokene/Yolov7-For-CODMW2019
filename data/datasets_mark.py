import os

path = r'./codmw2019/labels/train'
# print(len(os.listdir(path))) # 956
zero = open('zero.txt', 'w')
zero.truncate(0)
for item in os.listdir(path):
    path_item = os.path.join(path, item)
    with open(path_item, 'r') as f:
        list = f.readlines()
        if len(list) < 1:
            print(path_item)
            name = item[:-4] + '\n'
            zero.writelines(name)
            print('---')
zero.close()

image_ids = open('zero.txt').read().strip().split()
for image_id in image_ids:
    image_dir = "images/%s.png" % (image_id)
    xml_dir = "Annotations/%s.xml" % (image_id)
    if os.path.exists(image_dir):
        os.remove(image_dir)
        os.remove(xml_dir)
        print('成功删除文件:', image_dir)
    else:
        print('未找到此文件:', image_dir)
image_ids.close()