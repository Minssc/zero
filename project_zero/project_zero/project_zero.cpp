// project_zero.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "pch.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
void processImage(Mat); 
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|../../opencv401/cascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyes_cascade|../../opencv401/cascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nMade to test bunch of stuff for our project.\n\n");
	parser.printMessage();
	String face_cascade_name = parser.get<String>("face_cascade");
	String eyes_cascade_name = parser.get<String>("eyes_cascade");
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyes_cascade.load(eyes_cascade_name))
	{
		cout << "--(!)Error loading eyes cascade\n";
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
		int cam_dno = parser.get<int>("camera"); // cam device no 
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

			processImage(image);

			waitKey(0);
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
			processImage(image);
			waitKey(0); 
		}


	}
	return 0;
}

void processImage(Mat image) {

}
void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
		Mat faceROI = frame_gray(faces[i]);
		//-- In each face, detect eyes
		std::vector<Rect> eyes;
		eyes_cascade.detectMultiScale(faceROI, eyes);
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point eye_center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			circle(frame, eye_center, radius, Scalar(255, 0, 0), 4);
		}
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}