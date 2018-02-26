#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

// g++ -std=c++11 removeRedEyes.cpp `pkg-config --libs --cflags opencv` -o removeRedEyes

// ./removeRedEyes

using namespace cv;
using namespace std;

void resizeFace(Mat &face);
Mat detectFace(const Mat &image, CascadeClassifier &faceDetector);
void combineDataSets(const string& file1, const string& file2, const string& writeFile);

int main ()
{
	Mat face;
	Mat samples;
	vector<string> labels;
	CascadeClassifier faceDetector;

	// Load Ronaldo's Face
	Mat ronaldo = imread("Images/ronaldo.jpg", IMREAD_GRAYSCALE);
	imshow("Ronaldo", ronaldo);
	waitKey();
	face = detectFace(ronaldo, faceDetector); // get face sub-image
	resizeFace(face); // resize face to be 200 x 200 pixels
	imshow("Ronaldo Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1)); // Reshape matrix into row vector
	labels.push_back("Ronaldo");
	
	// Load Messi's Face
	Mat messi = imread("Images/messi.jpg", IMREAD_GRAYSCALE);
	imshow("Messi", messi);
	waitKey();
	face = detectFace(messi, faceDetector);
	resizeFace(face);
	imshow("Messi Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Messi");
	
	// Load Wilshere's Face
	Mat wilshere = imread("Images/wilshere.jpg", IMREAD_GRAYSCALE);
	imshow("Wilshere", wilshere);
	waitKey();
	face = detectFace(wilshere, faceDetector);
	resizeFace(face);
	imshow("Wilshere Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Wilshere");

	// Load Mustafi's Face
	Mat mustafi = imread("Images/mustafi.jpg", IMREAD_GRAYSCALE);
	imshow("Mustafi", mustafi);
	waitKey();
	face = detectFace(mustafi, faceDetector);
	resizeFace(face);
	imshow("Mustafi Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Mustafi");

	// Load Salah's Face
	Mat salah = imread("Images/salah.jpg", IMREAD_GRAYSCALE);
	imshow("Salah", salah);
	waitKey();
	face = detectFace(salah, faceDetector);
	resizeFace(face);
	imshow("Salah Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Salah");

	// cout << samples.rows << " : " << samples.cols << endl;

	// Save Data to .xml file

	FileStorage fs; // Create filestorage instance
	
	fs.open("10090379.xml", FileStorage::WRITE); // Filename is student number

	fs << "samples" << samples;
	fs << "labels" << labels;
	fs.release();

	// test combineDataSet function
	
	combineDataSets("10090379.xml", "10090379.xml", "combined.xml");

}

void resizeFace(Mat &face)
{
	resize(face, face, Size(200, 200)); // Resize face image to 200x200 pixels
}

Mat detectFace(const Mat &image, CascadeClassifier &faceDetector) 
{
	vector<Rect> faces;
	bool isok = faceDetector.load("/home/filipe/opencv/data/haarcascades/haarcascade_frontalface_default.xml");
	// cout << isok << endl;

	faceDetector.detectMultiScale(image, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30,30));

	if (faces.size() == 0)
	{
		cerr << "Error: No Faces Found" << endl;
		return Mat();
	}

	if (faces.size() > 1)
	{
		cerr << "Error: Multiple Faces Found" << endl;
	}
/*
	Mat detected = image.clone();
	for (unsigned int i = 0; i < faces.size(); i++) 
	{
		rectangle(detected, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0));
	}

	imshow("Faces Detected", detected);
	waitKey();
*/

	return image(faces[0]).clone();
}

void combineDataSets(const string& file1, const string& file2, const string& writeFile) {
	
	
	FileStorage fs; // create FileStorage instance
	Mat samples;
	Mat samples_temp;
	vector<string> labels;
	vector<string> labels_temp;
	
	fs.open(file1, FileStorage::READ);
	fs["samples"] >> samples;
	fs["labels"] >> labels;
	
	fs.open(file2, FileStorage::READ);
	fs["samples"] >> samples_temp;
	fs["labels"] >> labels_temp;

	samples.push_back(samples_temp); // combine two matrices

	labels.insert(labels.end(), labels_temp.begin(), labels_temp.end()); //combine labels

	fs.open(writeFile, FileStorage::WRITE);
	fs << "samples" << samples;
	fs << "labels" << labels;

	fs.release();

}
