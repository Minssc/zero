#include "pch.h" 
#include "improc.h"
#include <iostream> 
#include <stack> 

using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
#define DOFINDFACE 0 



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


	Point2i *searchNeighbour(Mat *image,Mat *labels,int x,int y) {
		Point2i *point = (Point2i*)malloc(sizeof(Point2i));
		point->x = -1, point->y = -1;
		int cx, cy;
		for (cx = x - 1; cx <= x+1; cx++) {
			for (cy = y - 1; cy <= y+1; cy++) {
				//_cx < 0 || _cx >= image->rows || _cy < 0 || _cy >= image->cols
				//image->at<uchar>(cx, cy) == 0 || labels->at<uchar>(cx, cy) != 0
				if (cx < 0 || cx >= image->rows || cy < 0 || cy >= image->cols)
					continue; 
				
				if (image->at<uchar>(cx, cy) == 0 || labels->at<uchar>(cx, cy) != 0)
					continue; 
				point->x = cx, point->y = cy;
				return point; 
			}
		}

		return point; 
	}

	void __sizeFilter(Mat *image,int min,int max) {
		Mat1b labels = Mat::zeros(image->size(), CV_8UC1);
		int x, y;
		int cx, cy;
		int area;
		int labelcolor = 255;
		std::stack<Point2i*> stack;

		std::stack<Point2i*> saved_stack; 

		for (x = 0; x < image->rows; x++) {
			for (y = 0; y < image->cols; y++) {
				if (image->at<uchar>(x, y) == 0 || labels.at<uchar>(x, y) != 0)
					continue; 
				cx = x, cy = y;
				area = 0; 
				//labelcolor = (labelcolor + 10) % 254 + 1; 

				while (1) {
				LOOP:
					Point2i *point = searchNeighbour(image, &labels, cx, cy);

					if (point->x > -1) { // neighbour found 
						area++;
						cx = point->x, cy = point->y;
						labels.at<uchar>(cx, cy) = labelcolor; 
						stack.push(point);
						goto LOOP;
					}
					else {
						if (stack.empty())
							break; // exit while loop 
						point = stack.top(); //get stack top
						cx = point->x, cy = point->y; 
						stack.pop(); // pop stack top 
						saved_stack.push(point);
						//free(point); 
						goto LOOP; // go back and try to find neighbour 
					}
				}

			//	cout << area << endl;
				while (1) {
					if (saved_stack.empty())
						break;
					Point2i *point = saved_stack.top();

					if (area < min || area >max) { // size filter apply 
						//labels.at<uchar>(point->x, point->y) = 0; 
						image->at<uchar>(point->x, point->y) = 0; 
					}

					saved_stack.pop(); 
					free(point); 
				}

			}
		}
		//labels.copyTo(*image);
		//imshow("LABELS", labels); 

	}

	void _sizeFilter(Mat *image) {
		int x, y;
		int cx, cy;
		int _cx, _cy;
		int area;
		int labelcolor = 100;
		stack <Point2i> stack; 
		Mat1b labels = Mat::zeros(image->size(), CV_8UC1);

		for (x = 0; x < image->rows/4; x++) {
			for (y = 0; y < image->cols/4; y++) {
				if (image->at<uchar>(x, y) == 0 || labels.at<uchar>(x, y) != 0)
					continue; 
				cx = x, cy = y, area = 1;
				labelcolor = (labelcolor + 10) % 254 + 1;
				cout << x << "," << y << endl; 

				while (1) {
GRASSFIRE:
					for (_cx = cx - 1; _cx <= cx + 1; _cx++) {
						for (_cy = cy - 1; _cy <= cy + 1; _cy++) {
							if (_cx < 0 || _cx >= image->rows || _cy < 0 || _cy >= image->cols) // ignore out of bounds
								continue; 

							if (image->at<uchar>(_cx, _cy) == 255 && labels.at<uchar>(_cx, _cy) == 0) {
								Point2i *point = (Point2i*)malloc(sizeof(Point2i)); 
								//cout << point << endl; 
								point->x = _cx, point->y = _cy;
								stack.push(*point); 
								labels.at<uchar>(_cx, _cy) = labelcolor;
								cx = _cx, cy = _cy, area++;
								goto GRASSFIRE;
							}

						}
					}
					if(stack.empty())
						break;
					Point2i point = stack.top(); 
					//cout << &point << endl;
					stack.pop(); 
				}


			}

		}

		imshow("LABELING", labels); 

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

	void balance_white(cv::Mat mat) {
		double discard_ratio = 0.05;
		int hists[3][256];
		memset(hists, 0, 3 * 256 * sizeof(int));

		for (int y = 0; y < mat.rows; ++y) {
			uchar* ptr = mat.ptr<uchar>(y);
			for (int x = 0; x < mat.cols; ++x) {
				for (int j = 0; j < 3; ++j) {
					hists[j][ptr[x * 3 + j]] += 1;
				}
			}
		}

		// cumulative hist
		int total = mat.cols*mat.rows;
		int vmin[3], vmax[3];
		for (int i = 0; i < 3; ++i) {
			for (int j = 0; j < 255; ++j) {
				hists[i][j + 1] += hists[i][j];
			}
			vmin[i] = 0;
			vmax[i] = 255;
			while (hists[i][vmin[i]] < discard_ratio * total)
				vmin[i] += 1;
			while (hists[i][vmax[i]] > (1 - discard_ratio) * total)
				vmax[i] -= 1;
			if (vmax[i] < 255 - 1)
				vmax[i] += 1;
		}


		for (int y = 0; y < mat.rows; ++y) {
			uchar* ptr = mat.ptr<uchar>(y);
			for (int x = 0; x < mat.cols; ++x) {
				for (int j = 0; j < 3; ++j) {
					int val = ptr[x * 3 + j];
					if (val < vmin[j])
						val = vmin[j];
					if (val > vmax[j])
						val = vmax[j];
					ptr[x * 3 + j] = static_cast<uchar>((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
				}
			}
		}
	}


	void processImage(Mat image) {
		Mat orig;
		image.copyTo(orig); // backup imaage

		//imshow("ORIG", orig);
		balance_white(image);
		imshow("WB", orig); 
		
		if (DOFINDFACE) {
			int r = findFace(&image, image);
			switch (r) {
			case 0:
				cout << "Cannot find a face!" << endl;
				return;
			case 2:
				cout << "Too many faces!" << endl;
				return;
			}
		}
		
		//imshow("FACE", image); 
		
		Mat image_gray;
		cvtColor(image, image_gray, COLOR_BGR2GRAY); // gray image 
		//imshow("cvtColor_gray", image_gray); 

		Mat image_HSV;
		cvtColor(image, image_HSV, COLOR_BGR2HSV); // HSV image 
		//equalizeHist(image_gray, image_gray);
		//imshow("equalizeHist", image_gray);
		imshow("HSV",image_HSV); 

		for (int i = 0; i < 10; i++) {
			medianBlur(image_gray, image_gray, 3); // remove noise 
		}

		adaptiveThreshold(image_gray, image_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 9, 1);
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
		/*
		
		inRange(image_HSV, Scalar(0, 50, 100), Scalar(8, 200, 255), mask1); // mask 
		inRange(image_HSV, Scalar(172, 50, 100), Scalar(180, 200, 255), mask2); // mask2 
		*/
		inRange(image_HSV, Scalar(0, 75, 230), Scalar(10, 255, 255), mask1); // mask 
		inRange(image_HSV, Scalar(170, 75, 230), Scalar(180, 255, 255), mask2); // mask2 

		Mat1b mask = mask1 | mask2;

		//imshow("inRange_before", mask);

		Mat dilElement = getStructuringElement(MORPH_DILATE, { 3,3 });
		Mat erodElement = getStructuringElement(MORPH_ERODE, { 3,3 });

		dilate(mask, mask, dilElement);
		erode(mask, mask, erodElement);

		imshow("inRange", mask);


		mask = mask & image_gray;

		//imshow("COMBINED MASK", mask);

		__sizeFilter(&mask,10,150); 

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
			drawContours(drawing, contours, i, color, 1, 8, noArray(), 0, Point());
			//cout << "area " << i << ": " << contourArea(contours[i]) << endl; 
		}

		imshow("Contours", drawing);
	}

}
