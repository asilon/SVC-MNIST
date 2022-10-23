import pickle
import cv2
import joblib
import argparse


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"], cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    loaded_model = joblib.load('model/svm_model.pkl')
    scaler = pickle.load(open('model/scaler.pkl', 'rb'))

    nx, ny = image.shape
    sample = image.reshape((1, nx * ny))
    normalized_sample = scaler.transform(sample)

    result = loaded_model.predict(normalized_sample)

    print(result)