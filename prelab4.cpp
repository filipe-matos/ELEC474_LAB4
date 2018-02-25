#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

void resizeFace(Mat &face);
Mat detectFace(const Mat &image, CascadeClassifier &faceDetector);

int main ()
{
	Mat face;
	Mat samples;
	vector<string> labels;
	CascadeClassifier faceDetector;

	// Load Ronaldo's Face
	Mat ronaldo = imread("Images/ronaldo.jpg");
	imshow("Ronaldo", ronaldo);
	waitKey();
	face = detectFace(ronaldo, faceDetector); // get face sub-image
	resizeFace(face); // resize face to be 200 x 200 pixels
	imshow("Ronaldo Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1)); // Reshape matrix into row vector
	labels.push_back("Ronaldo");

	cout << face.rows << " : " << face.cols << endl;
	
	// Load Messi's Face
	Mat messi = imread("Images/messi.jpg");
	imshow("Messi", messi);
	waitKey();
	face = detectFace(messi, faceDetector);
	resizeFace(face);
	imshow("Messi Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Messi");

	cout << face.rows << " : " << face.cols << endl;
	
	// Load Wilshere's Face
	Mat wilshere = imread("Images/wilshere.jpg");
	imshow("Wilshere", wilshere);
	waitKey();
	face = detectFace(wilshere, faceDetector);
	resizeFace(face);
	imshow("Wilshere Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Wilshere");

	cout << face.rows << " : " << face.cols << endl;

	// Load Mustafi's Face
	Mat mustafi = imread("Images/mustafi.jpg");
	imshow("Mustafi", mustafi);
	waitKey();
	face = detectFace(mustafi, faceDetector);
	resizeFace(face);
	imshow("Mustafi Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Mustafi");
	
	cout << face.rows << " : " << face.cols << endl;


	// Load Salah's Face
	Mat salah = imread("Images/salah.jpg");
	imshow("Salah", salah);
	waitKey();
	face = detectFace(salah, faceDetector);
	resizeFace(face);
	imshow("Salah Face", face);
	waitKey();
	samples.push_back(face.clone().reshape(1,1));
	labels.push_back("Salah");

	
	cout << face.rows << " : " << face.cols << endl;
	cout << samples.rows << " : " << samples.cols << endl;

}

void resizeFace(Mat &face)
{
	resize(face, face, Size(200, 200)); // Resize face image to 200x200 pixels
}

Mat detectFace(const Mat &image, CascadeClassifier &faceDetector) 
{
	vector<Rect> faces;
	bool isok = faceDetector.load("/home/filipe/opencv/data/haarcascades/haarcascade_frontalface_default.xml");
	cout << isok << endl;

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
