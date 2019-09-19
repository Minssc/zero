#include "improc.h"
#include <iostream> 
#include <stack> 
#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include "httpcl.hpp" 
#include <unistd.h> 
//#include "drawlandmarks.h"



using namespace std;
using namespace cv;
using namespace cv::face;

#define DOFINDFACE 0 
//#define DBG 

CascadeClassifier face_cascade;
extern Ptr<Facemark> facemark; 

extern string PZ_updateStatus(string);
//http httpc("127.0.0.1", "8080"); 
extern http *httpcl; 

namespace IMPROC {

	enum facePolyNames {
		FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, NOSE, PHILTRUM, MOUTH, CHIN, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW
	};

	vector<vector<int>> facePolyPoints = {
		{1,18,19,20,21,22,23,24,25,26,27,17,70,69}, // additional points required
		{13,14,15,16,17,46,47,48,43,36,55}, // LC
		{1,2,3,4,5,49,32,40,41,42,37}, // RC 
		{32,33,34,35,36,71,28,72}, // NO *71 and 72 avg of 40&28 and 43&28
		{49,50,51,52,53,54,55,36,35,34,33,32}, // PH 
		{49,50,51,52,53,54,55,56,57,58,59,60}, // MO 
		{5,6,7,8,9,10,11,12,13,55,56,57,58,59,60,49}, // CH 
		{43,44,45,46,47,48}, // LE 
		{37,38,39,40,41,42}, // RE 
		{18,19,20,21,22}, //LEY 
		{23,24,25,26,27} //REY
	};

	void drawPolyline(
		Mat &im,
		const vector<Point2f> &landmarks,
		const vector<int> &facePolyPoints,
		Scalar &color,
		bool isClosed = false
	){
		// Gather all points between the start and end indices
		vector <Point> points;
		for (int i = 0; i < facePolyPoints.size(); i++){
			points.push_back(cv::Point(landmarks[facePolyPoints[i]-1].x, landmarks[facePolyPoints[i]-1].y));
		}
		// Draw polylines. 
		polylines(im, points, isClosed, color, 1, 16);
	}

	void drawfacePoly(Mat &im, vector<Point2f> &landmarks, int FACEPOLYENUM,Scalar color){
		drawPolyline(im, landmarks, facePolyPoints[FACEPOLYENUM],color,true);
	}

	int processFace(Mat *dest, Mat frame,vector<vector<Point2f>> &landmarks,Point &offset) {
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame_gray, faces);
		if (faces.size() == 0)
			return 0; 
		else if(faces.size() > 1)
			return 2;
		
		offset = Point(faces[0].x, faces[0].y); 
		float lowest_point = 0.0f;

		if (facemark->fit(frame, faces, landmarks)) {
			lowest_point = landmarks[0][0].y;
			for (int i = 0; i < landmarks[0].size(); i++) {
				if (lowest_point < landmarks[0][i].y)
					lowest_point = landmarks[0][i].y;
			}
		}
		// augment point 69 and 70 
		landmarks[0].push_back(Point2f(landmarks[0][0].x,faces[0].y+2)); // point 69, combination of point 1's x and top of RoI
		landmarks[0].push_back(Point2f(landmarks[0][16].x, faces[0].y+2)); // point 70, combination of point 17's x and top of RoI
		landmarks[0].push_back((landmarks[0][27]+landmarks[0][42])/2); // point 71, avg of 28& 43
		landmarks[0].push_back((landmarks[0][27] + landmarks[0][39]) / 2); // point 72, avg of 40 & 28

		faces[0].height = faces[0].height > (lowest_point - faces[0].y) ? faces[0].height : (int)(lowest_point - faces[0].y + 3.5); // some additional pixels
			//cout << faces[0] << endl; 
		if (faces[0].height + faces[0].y >= frame.rows)
			return 3;
			//cout << faces[0].height + faces[0].y << ","<<frame.rows<<" passed" << endl;
		Mat iCrop = frame(faces[0]); 
		*dest = iCrop;
		
		return 1;
	}

	Point2i *searchNeighbour(Mat *image,Mat *labels,int x,int y) {
		Point2i *point = (Point2i*)malloc(sizeof(Point2i));
		point->x = -1, point->y = -1;
		int cx, cy;
		for (cx = x - 1; cx <= x+1; cx++) {
			for (cy = y - 1; cy <= y+1; cy++) {
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

	void sizeFilter(Mat *image,int min,int max) {
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
						goto LOOP; // go back and try to find neighbour 
					}
				}
				while (1) {
					if (saved_stack.empty())
						break;
					Point2i *point = saved_stack.top();

					if (area < min || area > max) { // size filter apply 
						image->at<uchar>(point->x, point->y) = 0; 
					}

					saved_stack.pop(); 
					free(point); 
				}

			}
		}
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

	void offsetContour(vector<Point> &contours, Point offset) {
		for (int i = 0; i < contours.size(); i++)
			contours[i] += offset; 
	}

	void scaleContour(vector<Point> &contours, double scale) {
		Rect crect = boundingRect(contours);
		Point offset = Point(crect.x, crect.y); // top left corner of bounding box 
		
		for (int i = 0; i < contours.size(); i++) // offset then scale then offset again all points
			contours[i] = ((contours[i] - offset) * scale) + (offset + Point((crect.width - crect.width * scale)/2,(crect.height - crect.height * scale)/2)); 
	}

	void scaleContour(vector<Point> &contours, double scale_x, double scale_y) {
		Rect crect = boundingRect(contours);
		Point offset = Point(crect.x, crect.y); // top left corner of bounding box 

		for (int i = 0; i < contours.size(); i++) { // offset then scale then offset again all points
			contours[i] -= offset; // offset the points 
			contours[i].x *= scale_x;
			contours[i].y *= scale_y;
			contours[i] += (offset + Point((crect.width - crect.width*scale_x) / 2, (crect.height - crect.height* scale_y) / 2));
		}

	}

	void divideFace(Mat image, vector<vector<Point2f>> &landmarks,vector<vector<Point>> &contours) {

		Mat1b board = Mat::zeros(image.size(), CV_8UC1);
		Mat board2;// = Mat::zeros(image.size(), CV_8UC3);
		board2 = image.clone(); 

		for (int FACEPOLYINDEX = 0; FACEPOLYINDEX < facePolyPoints.size(); FACEPOLYINDEX++) {
			board = Mat::zeros(image.size(), CV_8UC1);
			drawfacePoly(board, landmarks[0], FACEPOLYINDEX, Scalar(255, 255, 255));

			vector<vector<Point>> contours_temp;
			findContours(board, contours_temp, RETR_LIST, CHAIN_APPROX_NONE, Point(0, 0));
			//cout << "# of contours found: " << contours_temp.size() << endl; 

			double max = 0;
			for (int i = 0; i < contours_temp.size(); i++) {
				if (contourArea(contours_temp[i]) > max) {
					max = contourArea(contours_temp[i]);
					if (contours.size() > FACEPOLYINDEX)
						contours.erase(contours.begin() + FACEPOLYINDEX);
					contours.insert(contours.begin() + FACEPOLYINDEX, contours_temp[i]); // pick the biggest contour of the found group
				}
			}
			//FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, NOSE, PHILTRUM, MOUTH, CHIN, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW
			int tval; 
			switch (FACEPOLYINDEX) {
			case LEFT_CHEEK: case RIGHT_CHEEK:
				scaleContour(contours[FACEPOLYINDEX], 0.85);
				break;
			case PHILTRUM:
				scaleContour(contours[FACEPOLYINDEX], 1,0.6);
				break;
			case MOUTH:
				scaleContour(contours[FACEPOLYINDEX], 1.1);
				break;
			case NOSE:
				scaleContour(contours[FACEPOLYINDEX], 1.1, 0.90);
				break; 
			case FOREHEAD:
				tval = contours[FACEPOLYINDEX][0].y;
				scaleContour(contours[FACEPOLYINDEX], 1.00, 0.9);
				offsetContour(contours[FACEPOLYINDEX], Point(0, tval - contours[FACEPOLYINDEX][0].y));
				break;  
			case CHIN:
				tval = contours[FACEPOLYINDEX][8].y;
				scaleContour(contours[FACEPOLYINDEX], 1.00, 0.75);
				offsetContour(contours[FACEPOLYINDEX], Point(0,contours[FACEPOLYINDEX][8].y - tval));
				break;
			case LEFT_EYE: case RIGHT_EYE:
				scaleContour(contours[FACEPOLYINDEX], 1.25);
				break;
			case LEFT_EYEBROW: case RIGHT_EYEBROW:
				scaleContour(contours[FACEPOLYINDEX], 1.1);
				break;
			}

			for (int i = 0; i < contours.size(); i++) {
				drawContours(board2, contours, i, Scalar(0, 255, 255), 1, 8, noArray(), 0, Point(0, 0));
			}
			//imshow("DRAWPOLY BOARD", board);
#ifdef DBG
			imshow("CONTOUR BOARD", board2);
#endif
		}
		//cout << "final # of contours: " << contours.size() << endl; 
	}

	void getHSVAverage(Mat HSV_image, Mat1b &mask, Scalar &avg_col, Scalar &deviation, vector<vector<Point>> contours) {
		Rect RoI;
		Mat AoI;

		//Mat1b points = Mat::zeros(HSV_image.size(), CV_8UC1);
		enum {
			H, S, V
		};
		double m[3] = { 2.5, 5.0, 5.0 };

		for (int i = 0; i < contours.size(); i++) {

			unsigned long sum[3] = { 0,0,0 };
			unsigned int count = 0;

			double var[3] = { 0.0,0.0,0.0 };

			if (i == LEFT_EYE || i == RIGHT_EYE || i == LEFT_EYEBROW || i == RIGHT_EYEBROW || i == MOUTH)
				continue;
			RoI = boundingRect(contours[i]);
			AoI = HSV_image(RoI);
			//cout << RoI << endl; 
			//imshow("AoI", AoI);

			for (int x = 0; x < AoI.cols; x++) {
				for (int y = 0; y < AoI.rows; y++) {
					//cout << pointPolygonTest(contours[0], Point(x + RoI.x, y + RoI.y),false) << endl; 
					Point P = Point(x + RoI.x, y + RoI.y);
					//points.at<uchar>(P) = 100;

					if (pointPolygonTest(contours[i], P, false)==1) {
						//points.at<uchar>(P) = 255;
						for (int hsv = H; hsv <= V; hsv++) {
							sum[hsv] += HSV_image.at<Vec3b>(P)[hsv];
						}
						count++;
					}
				}
			}

			for (int hsv = H; hsv <= V; hsv++)
				avg_col[hsv] = (double)sum[hsv] / count;

			//cout << "avghsv: " << avg_col << endl; 
			//avg_col = Scalar((double)sH / count, (double)sS / count, (double)sV / count);

			for (int x = 0; x < AoI.cols; x++) {
				for (int y = 0; y < AoI.rows; y++) {
					Point P = Point(x + RoI.x, y + RoI.y);

					if (pointPolygonTest(contours[i], P, false)==1) {
						for (int hsv = H; hsv <= V; hsv++)
							var[hsv] += pow(HSV_image.at<Vec3b>(P)[hsv]-avg_col[hsv], 2);
					}
				}
			}

			for (int hsv = H; hsv <= V; hsv++)
				deviation[hsv] = sqrt(var[hsv] / count);

			//cout << "deviation: " << deviation << endl; 

			//deviation = Scalar(sqrt(vH / count), sqrt(vS / count), sqrt(vV / count));

			for (int x = 0; x < AoI.cols; x++) {
				for (int y = 0; y < AoI.rows; y++) {
					Point P = Point(x + RoI.x, y + RoI.y);
					if (pointPolygonTest(contours[i], P, false) != 1)
						continue;
					Vec3b Pixel = HSV_image.at<Vec3b>(P); 
					for (int hsv = H; hsv <= V; hsv++) {
						if (Pixel[hsv] <= (avg_col[hsv] - deviation[hsv] * m[hsv]) || Pixel[hsv] >= (avg_col[hsv] + deviation[hsv] * m[hsv])) {
							mask.at<uchar>(P) = 255;
							break;
						}
					}
				}
			}

			//cout << "avg HSV for d=" << i << " = " << avg_col << ", dev = " << deviation << endl; 
		}
			//imshow("PROCESSED AREA", points); 
	}

	void landmarkPoints(vector<Point2f> &landmarks, vector<Point2f> &data,int index) {
		for (int i = 0; i < facePolyPoints[index].size(); i++) {
			data.push_back(landmarks[facePolyPoints[index][i]-1]);
		}
	}

	int getPlacement(vector<vector<Point>> &contour, int index, vector<vector<Point2f>> &landmarks,Size imgsize) {
		//Rect CR = boundingRect(contour[index]); 
		Mat contourMat = Mat::zeros(imgsize, CV_8UC1); 
		drawContours(contourMat, contour, index, Scalar(255, 255, 255), 1, 8, noArray(), 0, Point());
		//imshow("WAT", contourMat); 
		Moments M = moments(contourMat, false);
		Point2f center = Point2f(M.m10 / M.m00, M.m01 / M.m00);
		Mat visualize = Mat::zeros(imgsize, CV_8UC3);
		//drawfacePoly(visualize, landmarks[i], i, Scalar(255, 0, 0));

		for (int i = 0; i < facePolyPoints.size(); i++) {
			vector<Point2f> lp;
			landmarkPoints(landmarks[0], lp, i);
		//	cout << pointPolygonTest(lp, center, false) << endl;
			drawfacePoly(visualize, landmarks[0], i, Scalar(255, 0, 0));
			circle(visualize, center, 1, Scalar(0, 0, 255), 1, 8, 0); 
			//imshow("VISUALIZE", visualize); 
			//waitKey(1000); 
			if (pointPolygonTest(lp, center, false) == 1) {
				return i; 
			}
		}
		return -1; 
	}

	void processImage(Mat image) {
		PZ_updateStatus("IMPROC: Starting");
		Mat orig;
		image.copyTo(orig); // backup imaage

		//imshow("ORIG", orig);
		balance_white(image);
		//imshow("WB", image);

		vector<vector<Point2f>> landmarks;
		vector<vector<Point>> contours_facemark;
		Point offset; 
		
		int r = processFace(&image, image, landmarks, offset);
		switch (r) {
		case 0:
			cout << "Cannot find a face!" << endl;
			PZ_updateStatus("WRN: No face detected");
			return; 
		case 2:
			cout << "Too many faces!" << endl;
			PZ_updateStatus("WRN: Too many faces");
			return;
		case 3:
			cout << "Face too close!" << endl; 
			PZ_updateStatus("WRN: Face too close");
			return; 
		}

		float multiplier = 512.0f / image.size().width; 

		resize(image, image, Size(image.size().width*multiplier,image.size().height*multiplier), 0, 0, INTER_LINEAR);
		//imshow("resized crop", image);

		for (int i = 0; i < landmarks.size(); i++) {
			for (int j = 0; j < landmarks[i].size(); j++) {
				landmarks[i][j] = (landmarks[i][j] - (Point2f)offset) * multiplier;
			}
		}


		divideFace(image, landmarks, contours_facemark);

#ifdef DBG
		imshow("FACE", image); 
#endif

		Mat image_gray;
		cvtColor(image, image_gray, COLOR_BGR2GRAY); // gray image 
		//imshow("cvtColor_gray", image_gray); 

		Mat image_HSV;
		cvtColor(image, image_HSV, COLOR_BGR2HSV); // HSV image 
		equalizeHist(image_gray, image_gray);
		//imshow("equalizeHist", image_gray);
		//imshow("HSV",image_HSV); 
		/*
		for (int i = 0; i < 10; i++) {
			medianBlur(image_gray, image_gray, 3); // remove noise 
		}*/

		//Mat1b threshtest = Mat::zeros(image.size(), CV_8UC1);
		//Mat1b cannytest = Mat::zeros(image.size(), CV_8UC1);
		/*
		for (int i = 3; i < 32; i += 2) {
			for (int k = 8; k < 13; k++) {
				double C = (double)k / 2;
				char str[512];
				adaptiveThreshold(image_gray, threshtest, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, i, C);
				sprintf_s(str, "TH %d %f\n", i, C);
				imshow(str, threshtest);
			}
		}
		*/
		medianBlur(image_gray, image_gray, 3);
		//GaussianBlur(image_gray, image_gray, Size(3, 3), 0);

		/*
		imshow("BLURRED", image_gray);
		Canny(image_gray, cannytest, 50, 150);
		imshow("CANNY", cannytest);
		Mat dilElement = getStructuringElement(MORPH_DILATE, { 7,7 });
		Mat erodElement = getStructuringElement(MORPH_ERODE, { 7,7 });

		dilate(cannytest, cannytest, dilElement);
		erode(cannytest, cannytest, erodElement);
		imshow("D-E CAN", cannytest);

		cout << image.size() << endl;
		*/
		adaptiveThreshold(image_gray, image_gray, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 9, 5);
		sizeFilter(&image_gray, 10, 65536); 
		//imshow("THRESHOLD", image_gray); 

		for (int i = 0; i < 10; i++) {
			medianBlur(image_gray, image_gray, 3); // remove noise 
		}

		Scalar avg_color;
		Scalar deviation;
		Mat1b FMmask = Mat::zeros(image.size(),CV_8UC1);

		getHSVAverage(image_HSV, FMmask, avg_color, deviation, contours_facemark);
		//cout << avg_color << endl; 
		//cout << deviation << endl; 
		//imshow("FM MASK", FMmask);

	//	Mat1b test1 = Mat::zeros(image.size(), CV_8UC1); 
		Mat1b mask = Mat::zeros(image.size(), CV_8UC1); 
	//	inRange(image_HSV, avg_color - deviation, avg_color + deviation, test1);

	//	imshow("MEAN DEV INRAGE", test1); 

		/*
		Mat1b mask1 = Mat::zeros(image.size(), CV_8UC1);
		Mat1b mask2 = Mat::zeros(image.size(), CV_8UC1);
		inRange(image_HSV, Scalar(0, 75, 230), Scalar(10, 255, 255), mask1); // mask 
		inRange(image_HSV, Scalar(170, 75, 230), Scalar(180, 255, 255), mask2); // mask2 

		Mat1b mask = mask1 | mask2;

		//imshow("inRange_before", mask);

		Mat dilElement = getStructuringElement(MORPH_DILATE, { 3,3 });
		Mat erodElement = getStructuringElement(MORPH_ERODE, { 3,3 });

		dilate(mask, mask, dilElement);
		erode(mask, mask, erodElement);
		*/
		//imshow("inRange", mask);

		Mat dilElement = getStructuringElement(MORPH_DILATE, { 3,3 });
		Mat erodElement = getStructuringElement(MORPH_ERODE, { 3,3 });

		//dilate(mask, mask, dilElement);
		//erode(mask, mask, erodElement);

		mask = FMmask & image_gray;



		// fill empty holes
		Mat ff = mask.clone();
		floodFill(ff, Point(0, 0),Scalar(255));
		Mat ffi = ~ff;
		
		//bitwise_not(ff,ffi); 

		mask = mask | ffi;
		
		sizeFilter(&mask,35,350); 
#ifdef DBG
		imshow("COMBINED MASK", mask);
#endif
		// end fill holes 
		//imshow("FF", result);

		//imshow("COMBINED and size filtered MASK", mask);

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

		vector<vector<Point>> contours_pimple;

		findContours(mask, contours_pimple, RETR_LIST, CHAIN_APPROX_NONE, Point(0, 0));

		Mat drawing = Mat::zeros(image_gray.size(), CV_8UC3);
		//cout << contours_pimple.size() << endl; 
		
		int spot; 
		
		for (int i = 0; i < contours_pimple.size(); i++){
			Scalar color = Scalar(0, 0, 255);
			if (((spot = getPlacement(contours_pimple, i, landmarks, image.size())) == NOSE) || (spot == -1))
				continue; //ignore nose for now
			drawContours(drawing, contours_pimple, i, color, 1, 8, noArray(), 0, Point());
			
			httpcl->start();
			httpcl->addHeader("pzero_result", to_string(spot));
			httpcl->get(HTTP_REQ); 
		//	httpc.start(); 
		//	httpc.addHeader("pzero_result", to_string(spot)); 
		//	httpc.get(HTTP_REQ); 
		//	PZ_updateStatus("IMPROC_RST: " + to_string(spot)); 
			//cout<<getPlacement(contours_pimple, i, landmarks,image.size())<<endl; 
		}
		
#ifdef DBG
		imshow("Pimple Contours", drawing);
#endif
		PZ_updateStatus("IMPROC: IMPROC FIN");
		//sleep(10); 
		for(int i=0; i<10; i++){
			waitKey(1000);
			PZ_updateStatus("IMPROC: Waiting"); 
		}
	}
}

