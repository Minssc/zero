// project_zero.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "pch.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace std;
using namespace cv;

void processImage(Mat); 

CascadeClassifier face_cascade;

int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{face_cascade|../../opencv401/cascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		//"{eyes_cascade|../../opencv401/cascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nMade to test bunch of stuff for our project.\n\n");
	parser.printMessage();
	String face_cascade_name = parser.get<String>("face_cascade");
	//String eyes_cascade_name = parser.get<String>("eyes_cascade");
	//-- 1. Load the cascades
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
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

int findFace(Mat *dest,Mat frame) {
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	vector<Rect> faces;
	face_cascade.detectMultiScale(frame_gray, faces);
	//imshow("Capture - Face detection", frame);
	if (faces.size() > 0) {
		Mat iCrop = frame(faces[0]); // only return the first face. 
		//imshow("ICROP", iCrop); 
		*dest = iCrop;
		return 1; 
	}
	return 0; 
}

void processImage(Mat image) {
	Mat orig;
	image.copyTo(orig); // backup imaage
	if (!findFace(&image, image)) {
		cout << "Cannot find a face!" << endl;
	}
	imshow("CROPPED FACE", image);

	Mat image_gray;
	cvtColor(image, image_gray, COLOR_BGR2GRAY); // gray image 

	Mat image_HSV;
	cvtColor(image, image_HSV, COLOR_BGR2HSV); // HSV image 

	//medianBlur(image_gray, image_gray, 11); // apply median blur to reduce noise 
	//GaussianBlur(image_gray, image_gray, { 3,3 } , 0); // gaussian blur 
	equalizeHist(image_gray, image_gray);

	//imshow("BLUR", image_gray);

	//threshold(image_gray, image_gray, 50, 255, THRESH_BINARY); 
	//Mat gaussian;
	//Mat mean;

	//adaptiveThreshold(image_gray, mean, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 2);
	adaptiveThreshold(image_gray, image_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
	//imshow("GAUSSIAN", gaussian);
	//imshow("mean", mean);

	for (int i = 0; i < 10; i++){
		medianBlur(image_gray, image_gray, 3); // remove noise 
		//medianBlur(gaussian, gaussian, 3); // remove noise 
		//medianBlur(mean, mean, 3); // remove noise 
	}
	//imshow("MEDIAN BLUR", image_gray);

	/*
	Mat dilElement = getStructuringElement(MORPH_DILATE, { 3,3 });
	Mat erodElement = getStructuringElement(MORPH_ERODE, { 3,3 });
	
	dilate(image_gray, image_gray, dilElement);
	dilate(image_gray, image_gray, dilElement);
	dilate(image_gray, image_gray, dilElement);
	erode(image_gray, image_gray, erodElement);
	erode(image_gray, image_gray, erodElement);
	erode(image_gray, image_gray, erodElement);
	*/
	imshow("END PRODUCT", image_gray);
	//imshow("END PRODUCT-g", gaussian);
	//imshow("END PRODUCT-m", mean);
	//Canny(image_gray, image_gray, 150, 200); 
	//imshow("CANNY", image_gray); 
	
	Mat image_masked = image.clone(); 
	for (int r = 0; r < image.rows; r++) { // apply mask to image 
		for (int c = 0; c < image.cols; c++) {
			Vec3b P = image_masked.at<Vec3b>(r, c);

			if (image_gray.at<uchar>(r, c) == 255)
				continue; 
			P[0] = 0; P[1] = 0; P[2] = 0; 
			image_masked.at<Vec3b>(r, c) = P; 
		}
	}

	imshow("MASKED IMAGE", image_masked); 


	Mat1b mask1 = Mat::zeros(image.size(), CV_8UC1);
	Mat1b mask2 = Mat::zeros(image.size(), CV_8UC1);

	inRange(image_HSV, Scalar(0, 50, 100), Scalar(6, 200, 255), mask1); // mask 
	inRange(image_HSV, Scalar(174, 50, 100), Scalar(180, 200, 255), mask2); // mask2 

	Mat1b mask = mask1 | mask2;

	imshow("COLOR MASK", mask);

	mask = mask & image_gray; 

	imshow("COMBINED MASK", mask); 
	vector<vector<Point>> contours; 

	//findContours(image_gray, contours, RETR_FLOODFILL, CHAIN_APPROX_NONE); 
	//drawContours(image_gray, contours, 0, { 255,255,255 }, 2, 8); 
}
