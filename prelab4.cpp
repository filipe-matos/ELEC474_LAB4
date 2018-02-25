
Mat detectFace(const Mat &image, CascadeClassifier &faceDetector);

int main ()
{
	Mat face;
	Mat samples;
	
	face = detectFace(image, faceDetector);
	resizeFace(face);
	samples.push_back(face.clone().reshape(1,1));

}

void resizeFace(Mat &face)
{
	resize(face, face, Size(200, 200)); // Resize face image to 200x200 pixels
}

Mat detectFace(const Mat &image, CascadeClassifier &faceDetector) 
{
	vector<Rect> faces;
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

	Mat detected = image.clone();
	for (unsigned int i = 0; i < faces.size(); i++) 
	{
		rectangle(detected, faces[i].tl(), faces[i].br(), Scalar(255, 0, 0));
	}

	imshow("Faces Detected", detected);
	waitKey();

	return image(faces[0]).clone();
}
