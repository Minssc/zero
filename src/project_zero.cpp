//#include "pch.h"
#include <iostream>
#include "improc.h"
#include "opencv2/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include <string.h>
#include <unistd.h> 
#include "httpcl.hpp" 
 
#define FACE_CASCADE_NAME "data/lbpcascades/lbpcascade_frontalface.xml" //"data/haarcascades/haarcascade_frontalface_default.xml"
#define FACEMARK_MODEL "data/facemark/lbfmodel.yaml"
#define CAM_DNO 0


//#define DBG

using namespace std;
using namespace cv;

//void processImage(Mat); 

extern CascadeClassifier face_cascade;
Ptr<face::Facemark> facemark = face::FacemarkLBF::create();
http *httpcl = new http("127.0.0.1", "8080"); 

string PZ_updateStatus(string status){
	cout << status << endl; 
	httpcl->start();
	httpcl->addHeader("pzero_status", status);
	return httpcl->get(HTTP_REQ); 
}


int main(int argc, const char *argv[])
{
	cout << "sending http request" << endl; 
	
	httpcl->start(); 
	httpcl->addHeader("pzero_initialize", ""); 
	httpcl->get("/testmod"); 
	//-- 1. Load the cascades

	PZ_updateStatus("Loading Face Cascade");
	if (!face_cascade.load(FACE_CASCADE_NAME)){
		PZ_updateStatus("ERR: Error loading face cascade");
		return -1;
	};
	

	PZ_updateStatus("Loading Facemark Model");
	facemark->loadModel(FACEMARK_MODEL);

	int capturemode = 0;
	
	if(argc > 1)
		if(strcmp(argv[1],"dev")==0)
			capturemode = 1; 

	Mat image; 
	Mat frame[5]; 
	PZ_updateStatus("Done Loading");

	if (capturemode == 0) {
		int cam_dno = CAM_DNO; // cam device no 
		VideoCapture c;
		c.open(cam_dno);


		if (!c.isOpened()) {
			PZ_updateStatus("ERR: Error opening video capture");
			return -1;
		}
		c.set(CAP_PROP_FRAME_WIDTH, 1280);
		c.set(CAP_PROP_FRAME_HEIGHT, 1024); 
		/*	
		cout<< c.get(CAP_PROP_BRIGHTNESS)<<endl;
		cout<< c.get(CAP_PROP_CONTRAST) << endl;
		cout<< c.get(CAP_PROP_SATURATION) << endl;
		cout<< c.get(CAP_PROP_GAIN) << endl; 
		*/
		cout<< c.get(CAP_PROP_EXPOSURE) << endl; 
		c.set(CAP_PROP_BRIGHTNESS, 0); 
		c.set(CAP_PROP_CONTRAST, 50);
		c.set(CAP_PROP_SATURATION, 55);
		c.set(CAP_PROP_GAIN, 64);
 		
	
		while(1){	
			
			c.read(image); 
			if(image.empty()){
				PZ_updateStatus("ERR: No captured frame");
				break; 
			}
			
			//int cf = 0; 
			//c.read(frame[0]); 

		//	flip(image,image,1); 
			Mat camview;
			resize(image, camview, Size(480,360), 0, 0, INTER_LINEAR); 	
#ifdef DBG
			imshow("IMG",camview); 	
#endif
			vector<Rect> faces;
			face_cascade.detectMultiScale(image, faces);

			//waitKey(1);
			//continue; 
			if(faces.size() == 0){
				PZ_updateStatus("WRN: No face detected");
				//sleep(1);
				waitKey(100); 
				continue; 
			}
			if(faces.size() > 1){
				PZ_updateStatus("WRN: Too many faces");
				waitKey(50); 
				continue;
			}
				
			if(faces[0].width*faces[0].height < 300*300){
				PZ_updateStatus("WRN: Face too far");
				waitKey(50); 
				continue; 
			}
			
			IMPROC::processImage(image);
			for(int i=0; i<5; i++)
				c.grab(); 
		}
	}
	else if (capturemode == 1) {
		string exts[] = { ".jpg",".png",".bmp","jpeg" }; // auto try extensions 
		string filename;

		while (1) {
			cout << "Input filename: ";
			cin >> filename;
			image = imread("samples/" + filename); // simpler loading image 

			if (!image.data) {
				for (int i = 0; i < sizeof(exts) / sizeof(exts[0]); i++) {
					image = imread("samples/" + filename + exts[i]);
					if (image.data)
						break;
				}
				if (!image.data) {
					cout << "Cannot find file: " << filename << endl;
					continue;
				}
			}
			IMPROC::processImage(image);
			waitKey(1000);
		}
	}
	return 0;
}
