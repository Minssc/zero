#ifndef IMPROC_H
#define IMPROC_H

#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"

using namespace cv;
namespace IMPROC {
	void printFacePolys();
	void processImage(Mat); 
};

#endif