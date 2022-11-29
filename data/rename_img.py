# -*- coding: utf-8 -*-

import os

class ImageRename():
    def __init__(self):
        self.path = './codmw2019/images/val'
    
    def rename(self):
        filelist = os.listdir(self.path)
        tot_num = len(filelist)
        
        i = 0
        for item in filelist:
            img_id = '%06d' % (i + 1)
            if item.endswith('.png'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), img_id + '.png')
                os.rename(src, dst)
                print('converting {} to {} ...'.format(src, dst))
                i = i + 1
        print('total {} to rename & converted {} imgs'.format(tot_num, i))


if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()