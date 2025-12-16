import os

ROOT = r"C:/Users/haris"
MRL_ROOT = ROOT + r"/mrlEyes/mrlEyes_2018_01"

subjects = []
all_images = []

for subject in os.listdir(MRL_ROOT):
  subject_dir = MRL_ROOT + r"/" + subject
  if("s00" not in subject): # Ignore annotations.txt and stats file
    continue

  subjects.append(subject)

  for eye in os.listdir(subject_dir):
     eye_state = int(eye.split("_")[4]) # 0:Closed, 1:Open
     eye_subject = subject
     eye_path = subject_dir + r"/" + eye
     
     eye_img = {"subject":eye_subject, "label":eye_state, "path":eye_path}
     all_images.append(eye_img)


# Report on dataset
print("Total number of subjects: ",len(subjects))
print("Total number of images: ",len(all_images))

open_eyes_per_subject = [0] * len(subjects)
closed_eyes_per_subject = [0] * len(subjects)

for img in all_images:
  idx = subjects.index(img['subject'])
  if(img['label'] == 0):
    closed_eyes_per_subject[idx] += 1
  elif(img['label'] == 1):
    open_eyes_per_subject[idx] += 1

for i in range(len(subjects)):
  print("\n Subject: ",subjects[i])
  print("Open eye images: ",open_eyes_per_subject[i])
  print("Closed eye images: ",closed_eyes_per_subject[i])



  

