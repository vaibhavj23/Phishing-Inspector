
import joblib

import sys
import numpy as np
import initial


def get_prediction_from_url(test_url):
    features_test = initial.main(test_url)
    
    return features_test 
    


def main():
    url = sys.argv[1]
    print(url)
    prediction = get_prediction_from_url(url)

    if prediction == 0:
        # print "The website is safe"
        print("SAFE ")
        
    elif prediction == 1:
        # print "The website is phishing"
        print("PHISHING")
        
    elif prediction == -1:
        # print "The website is suspicious"
        print("SUSPICIOUS")


if __name__ == "__main__":
    main()


