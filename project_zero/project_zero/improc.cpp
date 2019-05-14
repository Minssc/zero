#include "pch.h" 
#include "improc.h"
#include <iostream> 
#include <stack> 

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;

namespace IMPROC {
	int findFace(Mat *dest, Mat frame) {
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame_gray, faces);
		//imshow("Capture - Face detection", frame);
		if (faces.size() > 1) {
			return 2; // too many faces
		}else if(faces.size()==1){
			Mat iCrop = frame(faces[0]); 
			imshow("ICROP", iCrop);
			*dest = iCrop;
			return 1;
		}
		return 0;
	}

	void getArea(int x,int y,Mat* result) {

	}

	void sizeFilter(Mat *image) {
		int cx,cy,_cx,_cy,__cx,__cy;
		int area = 0; 
		int color = 55; 
		stack <Point2i> s;
		Mat1b map = Mat::zeros(image->size(), CV_8UC1);
		Mat1b mask = Mat::zeros(image->size(), CV_8UC1);

		for (cx=0; cx < image->rows; cx++) {
			for (cy = 0; cy < image->cols; cy++) {
				if (image->at<uchar>(cx, cy) == 255 || map.at<uchar>(cx,cy)!=0) // start counting pixels 
					continue;

				_cx = cx,_cy = cy; 
				area = 0;
				color = (color + 1) % 254 + 1;

				while (1) {

LOOP:

					for (__cx = _cx - 1; __cx <= _cx + 1; __cx++) {
						for (__cy = _cy - 1; __cy <= _cy + 1; __cy++) {
							if (__cx<0 || __cx >= image->rows || __cy <0 || __cy >= image->cols) // ignore out of bounds
								continue; 

							if (image->at<uchar>(__cx, __cy) == 255 && map.at<uchar>(__cx,__cy)==0) { // pos is set 
								Point2i *pixel = (Point2i*)malloc(sizeof(Point2i));
								pixel->x = __cy; pixel->y = __cx;
								_cx = __cx; _cy = __cy; area++;
								map.at<uchar>(__cx, __cy) = color;
								s.push(*pixel);
								goto LOOP; 
							}
						}
					}
					if (s.empty())
						break;
					if (area < 10 || area > 80) { // filter off too small or too big 
						Point2i pixel = s.top();
						mask.at<uchar>(pixel) = 255;
						s.pop();
						//free(&pixel); 
						continue;
						//cout << area << " is being removed" << endl; 
					}
					s.pop();
				}
			}
		}
		imshow("MAP", map); 
		*image = *image ^ mask; 

		imshow("SIZEFILTER MASK", mask); 
	}

	void processImage(Mat image) {
		Mat orig;
		image.copyTo(orig); // backup imaage
		//imshow("ORIG", orig); 
		
		int r = findFace(&image, image); 
		
		switch (r) {
		case 0:
			cout << "Cannot find a face!" << endl; 
			return; 
		case 2:
			cout << "Too many faces!" << endl; 
			return; 
		}
		
		//imshow("FACE", image); 
		
		Mat image_gray;
		cvtColor(image, image_gray, COLOR_BGR2GRAY); // gray image 
		//imshow("cvtColor_gray", image_gray); 

		Mat image_HSV;
		cvtColor(image, image_HSV, COLOR_BGR2HSV); // HSV image 
		equalizeHist(image_gray, image_gray);
		//imshow("equalizeHist", image_gray);


		adaptiveThreshold(image_gray, image_gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
		//imshow("THRESHOLD", image_gray); 

		for (int i = 0; i < 10; i++) {
			medianBlur(image_gray, image_gray, 3); // remove noise 
		}
		/*
		Mat dilElement = getStructuringElement(MORPH_DILATE, { 3,3 });
		Mat erodElement = getStructuringElement(MORPH_ERODE, { 3,3 });

		dilate(image_gray, image_gray, dilElement);
		erode(image_gray, image_gray, erodElement);
		*/

		Mat1b mask1 = Mat::zeros(image.size(), CV_8UC1);
		Mat1b mask2 = Mat::zeros(image.size(), CV_8UC1);

		inRange(image_HSV, Scalar(0, 50, 100), Scalar(8, 200, 255), mask1); // mask 
		inRange(image_HSV, Scalar(172, 50, 100), Scalar(180, 200, 255), mask2); // mask2 

		Mat1b mask = mask1 | mask2;
		//imshow("inRange", mask);


		mask = mask & image_gray;

		imshow("COMBINED MASK", mask);

		sizeFilter(&mask); 

		imshow("COMBINED and size filtered MASK", mask);

		Mat image_masked = image.clone();
		for (int r = 0; r < image.rows; r++) { // apply mask to image 
			for (int c = 0; c < image.cols; c++) {
				Vec3b P = image_masked.at<Vec3b>(r, c);

				if (mask.at<uchar>(r, c) == 255)
					continue;
				P[0] = 0; P[1] = 0; P[2] = 0;
				image_masked.at<Vec3b>(r, c) = P;
			}
		}

		//Mat erodElement = getStructuringElement(MORPH_ERODE, { 3,3 });

		//erode(image_masked, image_masked, erodElement);



		//imshow("FINAL MASKED IMAGE", image_masked);

		vector<vector<Point>> contours;

		findContours(mask, contours, RETR_LIST, CHAIN_APPROX_NONE, Point(0, 0));

		Mat drawing = Mat::zeros(image_gray.size(), CV_8UC3);
		for (int i = 0; i < contours.size(); i++)
		{
			/*
			int area = contourArea(contours[i]); 
			if (area <= 5 || area >= 100)
				continue; 
				*/
			Scalar color = Scalar(0, 255, 255);
			drawContours(drawing, contours, i, color, 2, 8, noArray(), 0, Point());
			//cout << "area " << i << ": " << contourArea(contours[i]) << endl; 
		}

		imshow("Contours", drawing);
	}

}
