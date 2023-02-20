import os
import pandas as pd
import numpy as np
import tkinter as tk
import cv2
import re
import shutil
import argparse
import torch
import torch.nn as nn
from torchvision.models import resnet152
from torchvision import transforms

from misc.utils import affine_transform, get_angle
from inference import SimpleHRNet, SimpleResNet152


def app(device, res_cpath):
    global resize_image, p, mv, index_store

    '''
    b: bounding box
    c: classify
    m: mSASSS scores
    o: original points
    p: plot points
    q: quit
    r: clear points
    s: edit mSASSS scores
    v: visibility checkbox
    '''

    index_store = []
    confidence_store = []
    class_idx_store = []

    class_idx_1 = ['3', 'no 3']
    class_idx_2 = ['']

    while(True):
        cv2.setMouseCallback(windowName, CallBackFunc)
        cv2.imshow(windowName, resize_image)
        
        key = cv2.waitKey(0)
        
        # Bounding Box
        if key == ord('b'):
            print('\nDisplay bounding boxes!\n')
            avg_len = abs(p[3] - p[5]).mean()
            for index, pair in enumerate(p):
                if index in index_store:
                    continue
                else:
                    if index % 2 == 1:
                        pt1 = (int(pair[0] - avg_len), int(pair[1] - avg_len))
                        pt2 = (int(pair[0] + avg_len), int(pair[1] + avg_len))
                        resize_image = cv2.rectangle(resize_image, pt1=pt1, pt2=pt2, color=(255, 255, 255), thickness=1)
                    else:
                        continue

        
        # Classify
        elif key == ord('c'):
            print('\nClassifying!\n')

            confidence_store.clear()
            class_idx_store.clear()
            p_copy = p.copy()
            m_reverse = cv2.getAffineTransform(src=dst, dst=src)
            for pair in p_copy:
                pair[:2] = affine_transform(pair, m_reverse, train=False)  

            # Crop
            assert 3 not in index_store and 5 not in index_store, 'Point 4 and Point 6 not visible for comptuation of y buffer!'
            avg_len = abs(p_copy[3] - p_copy[5]).mean()

            for index, pair in enumerate(p_copy):
                if index % 2 == 1 and index not in index_store:
                    # Rotation
                    pt1, pt2 = p_copy[index - 1][:2], pair[:2]
                    rot_deg = get_angle(pt1=pt1, pt2=pt2)
            
                    # Affine transform for each of the outer points
                    m = cv2.getRotationMatrix2D(center=(width // 2, height // 2), angle=rot_deg, scale=1)
                    image_rot = cv2.warpAffine(src=image, M=m, dsize=(width, height))
                    x, y = affine_transform(pt2, m, train=False)
                    rot_width, rot_height = image_rot.shape[1], image_rot.shape[0]

                    # Clip
                    low_pt = (np.clip(int(x - avg_len), a_min=0, a_max=rot_width), np.clip(int(y - avg_len), a_min=0, a_max=rot_height))
                    high_pt = (np.clip(int(x + avg_len), a_min=0, a_max=rot_width), np.clip(int(y + avg_len), a_min=0, a_max=rot_height))

                    # Classification: 3
                    image_crop = image_rot[low_pt[1]:high_pt[1] + 1, low_pt[0]:high_pt[0] + 1, :]

                    simpleresnet152 = SimpleResNet152(num_class=2, checkpoint_path=res_cpath, resolution=(224,224), device=torch.device(device))
                    idx, confidence = simpleresnet152.predict_single(image_crop)

                    print('Point {}:'.format(index + 1), class_idx_1[idx], '\t', 'Confidence:', confidence)

                    confidence_store.append('{:.4f}'.format(confidence))
                    class_idx_store.append('{}'.format(class_idx_1[idx]))

                
                elif index % 2 == 1 and index in index_store:
                    confidence_store.append('Not Applicable')
                    class_idx_store.append('Not Applicable')
                
            zip_list = list(zip(class_idx_store, confidence_store))

            print('\nDone\n')

        
        elif key == ord('m'):
            window = tk.Tk()
            window.title('mSASSS Scores')
            window.minsize(width=200, height=300)
            window.grid_columnconfigure((0, 1, 2), weight=1)

            tk.Label(master=window, text='Location').grid(row=0, column=0)
            tk.Label(master=window, text='Point 2 Upper').grid(row=1, column=0)
            tk.Label(master=window, text='Point 2 Lower').grid(row=2, column=0)
            tk.Label(master=window, text='Point 4 Upper').grid(row=3, column=0)
            tk.Label(master=window, text='Point 4 Lower').grid(row=4, column=0)
            tk.Label(master=window, text='Point 6 Upper').grid(row=5, column=0)
            tk.Label(master=window, text='Point 6 Lower').grid(row=6, column=0)
            tk.Label(master=window, text='Point 8 Upper').grid(row=7, column=0)
            tk.Label(master=window, text='Point 8 Lower').grid(row=8, column=0)
            tk.Label(master=window, text='Point 10 Upper').grid(row=9, column=0)
            tk.Label(master=window, text='Point 10 Lower').grid(row=10, column=0)
            tk.Label(master=window, text='Point 12 Upper').grid(row=11, column=0) 
            tk.Label(master=window, text='Point 12 Lower').grid(row=12, column=0)

            tk.Label(master=window, text='Class').grid(row=0, column=1)
            tk.Label(master=window, text=zip_list[0][0]).grid(row=1, column=1)
            tk.Label(master=window, text=zip_list[0][0]).grid(row=2, column=1)
            tk.Label(master=window, text=zip_list[1][0]).grid(row=3, column=1)
            tk.Label(master=window, text=zip_list[1][0]).grid(row=4, column=1)
            tk.Label(master=window, text=zip_list[2][0]).grid(row=5, column=1)
            tk.Label(master=window, text=zip_list[2][0]).grid(row=6, column=1)
            tk.Label(master=window, text=zip_list[3][0]).grid(row=7, column=1)
            tk.Label(master=window, text=zip_list[3][0]).grid(row=8, column=1)
            tk.Label(master=window, text=zip_list[4][0]).grid(row=9, column=1)
            tk.Label(master=window, text=zip_list[4][0]).grid(row=10, column=1)
            tk.Label(master=window, text=zip_list[5][0]).grid(row=11, column=1) 
            tk.Label(master=window, text=zip_list[5][0]).grid(row=12, column=1)

            tk.Label(master=window, text='Confidence').grid(row=0, column=2)
            tk.Label(master=window, text=zip_list[0][1]).grid(row=1, column=2)
            tk.Label(master=window, text=zip_list[0][1]).grid(row=2, column=2)
            tk.Label(master=window, text=zip_list[1][1]).grid(row=3, column=2)
            tk.Label(master=window, text=zip_list[1][1]).grid(row=4, column=2)
            tk.Label(master=window, text=zip_list[2][1]).grid(row=5, column=2)
            tk.Label(master=window, text=zip_list[2][1]).grid(row=6, column=2)
            tk.Label(master=window, text=zip_list[3][1]).grid(row=7, column=2)
            tk.Label(master=window, text=zip_list[3][1]).grid(row=8, column=2)
            tk.Label(master=window, text=zip_list[4][1]).grid(row=9, column=2)
            tk.Label(master=window, text=zip_list[4][1]).grid(row=10, column=2)
            tk.Label(master=window, text=zip_list[5][1]).grid(row=11, column=2) 
            tk.Label(master=window, text=zip_list[5][1]).grid(row=12, column=2)

            window.mainloop()


        # Plot Original Points
        elif key == ord('o'):
            print('\nPlot original points!\n')
            index_store = []
            p = p_original.copy()
            mv = mv_original.copy()
            resize_image = clone.copy()
            for index, pair in enumerate(p):
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
        
        
        # Plot Points
        elif key == ord('p'):
            print('\nPlot points!\n')
            for index, pair in enumerate(p):
                if index in index_store:
                    continue
                else:
                    resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                    resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))


        # Close
        elif key == ord('q'):
            print('\nQuiting!\n')
            break


        # Remove Points
        elif key == ord('r'):
            print('\nRemove points!\n')
            resize_image = clone.copy()


        # Save
        elif key == ord('s'):
            print('\nSaving!\n')
            if len(mSASSS) == 0:

                window = tk.Tk()
                window.title('Manual mSASSS Score')
                window.minsize(width=200, height=300)
                window.grid_columnconfigure((0, 1), weight=1) 

                tk.Label(master=window, text="Manual mSASSS Score").grid(row=0, columnspan=2, sticky='ew')
                tk.Label(master=window, text="Point 1").grid(row=1, column=0)
                entry1 = tk.Entry(master=window)
                entry1.grid(row=1, column=1)
                
                tk.Label(master=window, text="Point 2").grid(row=2, column=0)
                entry2 = tk.Entry(master=window)
                entry2.grid(row=2, column=1) 
                
                tk.Label(master=window, text="Point 3").grid(row=3, column=0)
                entry3 = tk.Entry(master=window)
                entry3.grid(row=3, column=1)           
                
                tk.Label(master=window, text="Point 4").grid(row=4, column=0)
                entry4 = tk.Entry(master=window)
                entry4.grid(row=4, column=1)
                
                tk.Label(master=window, text="Point 5").grid(row=5, column=0)
                entry5 = tk.Entry(master=window)
                entry5.grid(row=5, column=1)      
                
                tk.Label(master=window, text="Point 6").grid(row=6, column=0)
                entry6 = tk.Entry(master=window)
                entry6.grid(row=6, column=1)
                
                tk.Label(master=window, text="Point 7").grid(row=7, column=0)
                entry7 = tk.Entry(master=window)
                entry7.grid(row=7, column=1)
                
                tk.Label(master=window, text="Point 8").grid(row=8, column=0) 
                entry8 = tk.Entry(master=window)
                entry8.grid(row=8, column=1)
                
                tk.Label(master=window, text="Point 9").grid(row=9, column=0)
                entry9 = tk.Entry(master=window)
                entry9.grid(row=9, column=1)
                
                tk.Label(master=window, text="Point 10").grid(row=10, column=0)
                entry10 = tk.Entry(master=window)
                entry10.grid(row=10, column=1)
                
                tk.Label(master=window, text="Point 11").grid(row=11, column=0)
                entry11 = tk.Entry(master=window)
                entry11.grid(row=11, column=1)
                
                tk.Label(master=window, text="Point 12").grid(row=12, column=0)
                entry12 = tk.Entry(master=window)
                entry12.grid(row=12, column=1)

            else:
                window = tk.Tk()
                window.title('mSASSS Score')
                window.minsize(width=200, height=300)
                window.grid_columnconfigure(tuple(range(12)), weight=1)

                score_1 = mSASSS['score_1']
                score_2 = mSASSS['score_2']
                score_3 = mSASSS['score_3']
                score_4 = mSASSS['score_4']
                score_5 = mSASSS['score_5']
                score_6 = mSASSS['score_6']
                score_7 = mSASSS['score_7']
                score_8 = mSASSS['score_8']
                score_9 = mSASSS['score_9']
                score_10 = mSASSS['score_10']
                score_11 = mSASSS['score_11']
                score_12 = mSASSS['score_12']

                tk.Label(master=window, text="mSASSS Score").grid(row=0, columnspan=2, sticky='ew')
                tk.Label(master=window, text="Point 1").grid(row=1, column=0)
                entry1 = tk.Entry(master=window)
                entry1.insert(0, score_1)
                entry1.grid(row=1, column=1)
                
                tk.Label(master=window, text="Point 2").grid(row=2, column=0)
                entry2 = tk.Entry(master=window)
                entry2.insert(0, score_2)
                entry2.grid(row=2, column=1) 
                
                tk.Label(master=window, text="Point 3").grid(row=3, column=0)
                entry3 = tk.Entry(master=window)
                entry3.insert(0, score_3)
                entry3.grid(row=3, column=1)           
                
                tk.Label(master=window, text="Point 4").grid(row=4, column=0)
                entry4 = tk.Entry(master=window)
                entry4.insert(0, score_4)
                entry4.grid(row=4, column=1)
                
                tk.Label(master=window, text="Point 5").grid(row=5, column=0)
                entry5 = tk.Entry(master=window)
                entry5.insert(0, score_5)
                entry5.grid(row=5, column=1)      
                
                tk.Label(master=window, text="Point 6").grid(row=6, column=0)
                entry6 = tk.Entry(master=window)
                entry6.insert(0, score_6)
                entry6.grid(row=6, column=1)
                
                tk.Label(master=window, text="Point 7").grid(row=7, column=0)
                entry7 = tk.Entry(master=window)
                entry7.insert(0, score_7)
                entry7.grid(row=7, column=1)
                
                tk.Label(master=window, text="Point 8").grid(row=8, column=0)
                entry8 = tk.Entry(master=window)
                entry8.insert(0, score_8)
                entry8.grid(row=8, column=1)
                
                tk.Label(master=window, text="Point 9").grid(row=9, column=0)
                entry9 = tk.Entry(master=window)
                entry9.insert(0, score_9)
                entry9.grid(row=9, column=1)
                
                tk.Label(master=window, text="Point 10").grid(row=10, column=0)
                entry10 = tk.Entry(master=window)
                entry10.insert(0, score_10)
                entry10.grid(row=10, column=1)
                
                tk.Label(master=window, text="Point 11").grid(row=11, column=0)
                entry11 = tk.Entry(master=window)
                entry11.insert(0, score_11)
                entry11.grid(row=11, column=1)
                
                tk.Label(master=window, text="Point 12").grid(row=12, column=0)
                entry12 = tk.Entry(master=window)
                entry12.insert(0, score_12)
                entry12.grid(row=12, column=1)

            # Function
            def record():
                global mSASSS

                mSASSS['score_1'] = entry1.get()
                mSASSS['score_2'] = entry2.get()
                mSASSS['score_3'] = entry3.get()
                mSASSS['score_4'] = entry4.get()
                mSASSS['score_5'] = entry5.get()
                mSASSS['score_6'] = entry6.get()
                mSASSS['score_7'] = entry7.get()
                mSASSS['score_8'] = entry8.get()
                mSASSS['score_9'] = entry9.get()
                mSASSS['score_10'] = entry10.get()
                mSASSS['score_11'] = entry11.get()
                mSASSS['score_12'] = entry12.get()
                return window.destroy()

            tk.Button(master=window, text="Submit", command=record).grid(row=13, columnspan=2)
            window.mainloop()


        # Select Visibility
        elif key == ord('v'):
            print('\nVisibility\n')
            window = tk.Tk()
            window.title('Visibility')
            window.minsize()

            tk.Label(master=window, text='Visibility').pack()

            def change_val():
                global index_store

                var_store = []
                index_store = []
                var_store.append(point_1.get())
                var_store.append(point_2.get())
                var_store.append(point_3.get())
                var_store.append(point_4.get())
                var_store.append(point_5.get())
                var_store.append(point_6.get())
                var_store.append(point_7.get())
                var_store.append(point_8.get())
                var_store.append(point_9.get())
                var_store.append(point_10.get())
                var_store.append(point_11.get())
                var_store.append(point_12.get())
                
                index_store = [index for (index, var) in enumerate(var_store) if var == 1]
                return

            def display():
                global resize_image

                resize_image = clone.copy()
                for index, pair in enumerate(p):
                    if index in index_store:
                        continue
                    else:
                        resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                        resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
                return cv2.imshow(windowName, resize_image)


            point_1 = tk.IntVar()
            point_2 = tk.IntVar()
            point_3 = tk.IntVar()
            point_4 = tk.IntVar()
            point_5 = tk.IntVar()
            point_6 = tk.IntVar()
            point_7 = tk.IntVar()
            point_8 = tk.IntVar()
            point_9 = tk.IntVar()
            point_10 = tk.IntVar()
            point_11 = tk.IntVar()
            point_12 = tk.IntVar()

            c1 = tk.Checkbutton(window, text='Point 1', variable=point_1, onvalue=1, offvalue=0, command=change_val)
            c1.pack()
            c2 = tk.Checkbutton(window, text='Point 2', variable=point_2, onvalue=1, offvalue=0, command=change_val)
            c2.pack()
            c3 = tk.Checkbutton(window, text='Point 3', variable=point_3, onvalue=1, offvalue=0, command=change_val)
            c3.pack()
            c4 = tk.Checkbutton(window, text='Point 4', variable=point_4, onvalue=1, offvalue=0, command=change_val)
            c4.pack()
            c5 = tk.Checkbutton(window, text='Point 5', variable=point_5, onvalue=1, offvalue=0, command=change_val)
            c5.pack()
            c6 = tk.Checkbutton(window, text='Point 6', variable=point_6, onvalue=1, offvalue=0, command=change_val)
            c6.pack()
            c7 = tk.Checkbutton(window, text='Point 7', variable=point_7, onvalue=1, offvalue=0, command=change_val)
            c7.pack()
            c8 = tk.Checkbutton(window, text='Point 8', variable=point_8, onvalue=1, offvalue=0, command=change_val)
            c8.pack()
            c9 = tk.Checkbutton(window, text='Point 9', variable=point_9, onvalue=1, offvalue=0, command=change_val)
            c9.pack()
            c10 = tk.Checkbutton(window, text='Point 10', variable=point_10, onvalue=1, offvalue=0, command=change_val)
            c10.pack()
            c11 = tk.Checkbutton(window, text='Point 11', variable=point_11, onvalue=1, offvalue=0, command=change_val)
            c11.pack()
            c12 = tk.Checkbutton(window, text='Point 12', variable=point_12, onvalue=1, offvalue=0, command=change_val)
            c12.pack()

            tk.Button(master=window, text='Submit', command=display).pack()

            window.mainloop()

    # Return to main window
    cv2.destroyAllWindows()

    print('\nNot visible points\n', [x+1 for x in index_store])

# Callback function
def CallBackFunc(event, x, y, flags, param):
    global row, p, resize_image

    # Left click to change position
    if event == cv2.EVENT_RBUTTONDOWN:
        # Tkinter
        window = tk.Tk()

        tk.Label(master=window, text='Enter Point Index').pack()
        entry = tk.Entry(master=window)
        entry.pack()

        def change_pos():
            global p, resize_image

            index = int(entry.get())
            p[index-1][:2] = [x, y]
            resize_image = clone.copy()
           
            for index, pair in enumerate(p):
                if index in index_store:
                    continue
                else:
                    resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                    resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))

            cv2.imshow(windowName, resize_image)
            return window.destroy()

        btn = tk.Button(master=window, text='Submit', command=change_pos)
        btn.pack()
        window.mainloop()

    # Drag and Drop
    if event == cv2.EVENT_LBUTTONDOWN:
        row = np.where(np.linalg.norm(p - np.array([x, y]), axis=1) < 10)[0]
        assert row.shape[0] == 1, 'There may be more than 1 or none index in array!' 
        row = row[0]

    if event == cv2.EVENT_LBUTTONUP:
        assert type(row) == np.int64, 'None selected!'
        p[row][:2] = [x, y]
        resize_image = clone.copy()
        row = None

        for index, pair in enumerate(p):
            if index in index_store:
                continue
            else:
                resize_image = cv2.circle(resize_image, (int(pair[0]), int(pair[1])), radius=5, color=(255, 255, 255), thickness=-1)
                resize_image = cv2.putText(resize_image, text=str(index+1), org=(int(pair[0] + 20), int(pair[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1., color=(255, 255, 255))
        cv2.imshow(windowName, resize_image)


def main(ipath, hr_cpath, res_cpath, device):
    global windowName, image, height, width, clone, resize_image, src, dst, p, mv, p_original, mv_original, mSASSS


    mSASSS = {}
    
    for image_name in os.listdir(ipath):
        print('\nCurrent File: {}\n'.format(image_name))

        # Image
        image = cv2.imread(os.path.join(ipath, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[0], image.shape[1]

        # Prediction
        simplehrnet = SimpleHRNet(c=48, key=12, checkpoint_path=hr_cpath, device=torch.device(device))
        p, mv = simplehrnet.predict_single(image)
        p = p[0]
        mv = mv[0]

        # Affine Transformation
        src = np.float32([[0, 0], [width, 0], [0, height]])
        dst = np.float32([[0, 0], [1000, 0], [0, 1000]])
        m = cv2.getAffineTransform(src=src, dst=dst)
        resize_image = cv2.warpAffine(image, M=m, dsize=(1000, 1000))

        for pair in p:
            pair[:2] = affine_transform(pair, m, train=False)

        # Cache   
        clone = resize_image.copy()
        p_original, mv_original = p.copy(), mv.copy()

        # CV2 Window
        windowName = "Keypoint Image"
        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, 500, 200)

        # Tkinter (GUI)
        app(device, res_cpath)               

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipath', '-ip', help='path to image folder', type=str,  default='datasets/COCO/default')
    parser.add_argument('--res_cpath', '-rcp', help='path to resnet checkpoint', type=str,  default='logs/020223_104428/checkpoint_best_acc_0.8055555555555556.pth')
    parser.add_argument('--hr_cpath', '-hcp', help='path to hrnet checkpoint', type=str,  default='logs/20221220_1651/checkpoint_best_acc_0.9928728138145647.pth')
    parser.add_argument('--device', '-d', help='device', type=str, default='cpu')
    args = parser.parse_args()

    main(**args.__dict__)