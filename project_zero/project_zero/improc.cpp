#include "pch.h" 
#include "improc.h"
#include <iostream> 
#include <stack> 
//#include "drawlandmarks.h"


using namespace std;
using namespace cv;
using namespace cv::face;

CascadeClassifier face_cascade;
#define DOFINDFACE 0 
extern Ptr<Facemark> facemark; 

namespace IMPROC {

	enum facePolyNames {
		FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, NOSE, PHILTRUM, MOUTH, CHIN, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW
	};

	std::vector<std::vector<int>> facePolyPoints = {
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

	int findFace(Mat *dest, Mat frame,vector<vector<Point2f>> &landmarks,Point &offset) {
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		//-- Detect faces
		vector<Rect> faces;
		face_cascade.detectMultiScale(frame_gray, faces);
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

		if (faces.size() > 1) {
			return 2; // too many faces
		}else if(faces.size()==1){
			cout << faces[0] << endl; 
			faces[0].height = faces[0].height > (lowest_point - faces[0].y) ? faces[0].height : (int)(lowest_point - faces[0].y + 2.5); // some additional pixels
			cout << faces[0] << endl; 
			Mat iCrop = frame(faces[0]); 
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

					if (area < min || area >max) { // size filter apply 
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

	void scaleContour(vector<vector<Point>> &contours,int index,float scale) {
		Rect crect = boundingRect(contours[index]);
		Point offset = Point(crect.x, crect.y); // top left corner of bounding box 
		
		for (int i = 0; i < contours[index].size(); i++) { // offset then scale then offset again all points
			contours[index][i] = ((contours[index][i] - offset) * scale) + (offset + Point((crect.width - crect.width * scale)/2,(crect.height - crect.height * scale)/2)); 
		}
	}

	void divideFace(Mat image, vector<vector<Point2f>> &landmarks,vector<vector<Point>> &contours) {

		Mat1b board = Mat::zeros(image.size(), CV_8UC1);
		Mat board2;// = Mat::zeros(image.size(), CV_8UC3);
		board2 = image.clone(); 
		waitKey(3000);

		for (int FACEPOLYINDEX = 0; FACEPOLYINDEX < facePolyPoints.size(); FACEPOLYINDEX++) {
			board = Mat::zeros(image.size(), CV_8UC1);
			drawfacePoly(board, landmarks[0], FACEPOLYINDEX, Scalar(255, 255, 255));

			vector<vector<Point>> contours_temp;
			findContours(board, contours_temp, RETR_LIST, CHAIN_APPROX_NONE, Point(0, 0));

			float max = 0;
			for (int i = 0; i < contours_temp.size(); i++) {
				if (contourArea(contours_temp[i]) > max) {
					max = contourArea(contours_temp[i]);
					if (contours.size() > FACEPOLYINDEX)
						contours.erase(contours.begin() + FACEPOLYINDEX);
					contours.insert(contours.begin() + FACEPOLYINDEX, contours_temp[i]); // pick the biggest contour of the found group
				}
			}
			//FOREHEAD, LEFT_CHEEK, RIGHT_CHEEK, NOSE, PHILTRUM, MOUTH, CHIN, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW
			switch (FACEPOLYINDEX) {
			case LEFT_CHEEK: case RIGHT_CHEEK: case PHILTRUM: 
				scaleContour(contours, FACEPOLYINDEX, 0.85);
				break;
			case LEFT_EYE: case RIGHT_EYE:
				scaleContour(contours, FACEPOLYINDEX, 1.25);
				break;
			}

			for (int i = 0; i < contours.size(); i++) {
				drawContours(board2, contours, i, Scalar(0, 255, 255), 1, 8, noArray(), 0, Point(0, 0));
			}
			imshow("DRAWPOLY BOARD", board);
			imshow("CONTOUR BOARD", board2);
			waitKey(1000); 

		}
		cout << "final # of contours: " << contours.size() << endl; 
	}
	void processImage(Mat image) {
		Mat orig;
		image.copyTo(orig); // backup imaage

		//imshow("ORIG", orig);
		balance_white(image);
		imshow("WB", orig); 

		vector<vector<Point2f>> landmarks;
		vector<vector<Point>> contours_facemark;
		Point offset; 
		
		int r = findFace(&image, image, landmarks, offset);
		switch (r) {
		case 0:
			cout << "Cannot find a face!" << endl;
			if (DOFINDFACE)
				return; 
		case 2:
			cout << "Too many faces!" << endl;
			return;
		}

		

		//cout << "image size:" << image.size() << endl; 
		float multiplier = 512.0f / image.size().width; 

		resize(image, image, Size(image.size().width*multiplier,image.size().height*multiplier), 0, 0, INTER_LINEAR);
		imshow("resized crop", image);
		//cout << "resized size:" << image.size() << endl;

		for (int i = 0; i < landmarks.size(); i++) {
			for (int j = 0; j < landmarks[i].size(); j++) {
				//cout << "before: " <<landmarks[i][j] << endl; 
				landmarks[i][j] = (landmarks[i][j] - (Point2f)offset) * multiplier;
				//cout << "after: " << landmarks[i][j] << endl;
			}
		}


		divideFace(image, landmarks, contours_facemark);

		//imshow("FACE", image); 
		
		Mat image_gray;
		cvtColor(image, image_gray, COLOR_BGR2GRAY); // gray image 
		//imshow("cvtColor_gray", image_gray); 

		Mat image_HSV;
		cvtColor(image, image_HSV, COLOR_BGR2HSV); // HSV image 
		equalizeHist(image_gray, image_gray);
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

		imshow("inRange", mask);


		mask = mask & image_gray;

		//imshow("COMBINED MASK", mask);

		sizeFilter(&mask,10,150); 

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

		vector<vector<Point>> contours_pimple;

		findContours(mask, contours_pimple, RETR_LIST, CHAIN_APPROX_NONE, Point(0, 0));

		Mat drawing = Mat::zeros(image_gray.size(), CV_8UC3);
		for (int i = 0; i < contours_pimple.size(); i++){
			Scalar color = Scalar(0, 255, 255);
			drawContours(drawing, contours_pimple, i, color, 1, 8, noArray(), 0, Point());
		}

		imshow("Pimple Contours", drawing);

	}
}

