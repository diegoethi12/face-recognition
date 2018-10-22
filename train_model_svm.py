# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import os
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings"
    , required=True
    , help="path to serialized db of facial embeddings"
    #, default ='\\embeddings\\version1'
    )
ap.add_argument("-r", "--recognizer"
    , required=True
	, help="path to output model trained to recognize faces"
    #, default ='recognize_face_svm.pickle'
    )
ap.add_argument("-l", "--le"
    , required=True
	, help="path to output label encoder"
    #, default ='le.pickle'
    )
args = vars(ap.parse_args())

# load the face embeddings
print("[INFO] loading face embeddings...")
embedding_paths = os.getcwd()+"\\"+args['embeddings']
data = pickle.loads(open(embedding_paths, "rb").read())
print(data)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
# write the label encoder to disk
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

print("[INFO] Finished")