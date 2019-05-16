// project_zero.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "pch.h"
#include <iostream>
#include "improc.h"

#define FACE_CASCADE_NAME "../../opencv401/cascades/haarcascade_frontalface_alt.xml"
#define CAM_DNO 0

using namespace std;
using namespace cv;

//void processImage(Mat); 

extern CascadeClassifier face_cascade;

int main(int argc, const char** argv)
{
	//-- 1. Load the cascades

	if (!face_cascade.load(FACE_CASCADE_NAME))
	{
		cout << "--(!)Error loading face cascade"<< endl;
		return -1;
	};

	int capturemode = -1;

	do { // Select whether we use webcam or image file as input. 
		cout << "Select input method(0:webc 1:image): " << endl;

		cin >> capturemode;

		if (capturemode < 0 || capturemode > 1)
			cout << "Please input a valid input method" << endl;
	} while (capturemode < 0 || capturemode > 1);

	Mat image;

	if (capturemode == 0) {
		int cam_dno = CAM_DNO; // cam device no 
		VideoCapture c;
		c.open(cam_dno);

		if (!c.isOpened()) {
			cout << "Error opening video capture" << endl;
			return -1;
		}

		while (c.read(image)) {
			if (image.empty()) {
				cout << "No captured frame" << endl;
				break;
			}
			IMPROC::processImage(image);
			waitKey(2000);
		}
	}
	else if (capturemode == 1) {
		String exts[] = { ".jpg",".png",".bmp"}; // auto detect extensions 
		String filename;

		while (1) {
			cout << "Input filename: "; 
			cin >> filename;
			image = imread("../../samples/" + filename); // simpler loading image 
		
			if (!image.data) {
				for (int i = 0; i < sizeof(exts) / sizeof(exts[0]); i++) {
					image = imread("../../samples/" + filename + exts[i]);
					if (image.data)
						break;
				}
				if (!image.data) {
					cout << "Cannot find file: " << filename << endl;
					continue; 
				}
			}
			IMPROC::processImage(image);
			waitKey(0); 
		}


	}
	return 0;
}