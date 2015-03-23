#include <cstdlib>
#include <iostream>
#include <cstring>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace cv ;
using namespace std ;

//Global variables.
String faceFilePath = "D:\\User Programs\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml" ;
String eyesFilePath = "D:\\User Programs\\opencv\\sources\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml" ;

//Function declarations.
Mat *faceDetection( Mat * , Mat * , int ) ;
Mat *eyesDetection( Mat * , Mat * , Rect * ) ;
void faceBlur( Mat * ) ;
void faceDilation( Mat * ) ;
void faceGaussianBlur( Mat * ) ;
void faceMedianBlur( Mat * ) ;
Mat *beforeAfterMerge( Mat * , Mat * , Mat * ) ;

//Main function.
int main( int argc , char** argv )
{
	char input , buffer[ 10 ] ;
	bool conti = false , errInput = false ;
	string imgPath , fileName ;
	int saved = 0 , methods = 0 ;

	cout << "Welcome to use my Image Processing Final Project." << endl ;
	cout << "This project is about Face Mosaics." << endl ;
	cout << "\n====================Program Start====================\n" << endl ;

	//Main process loop.
	do
	{
		Mat image , resultImg , mergeImg ;
		Mat *resultPtr , *mergePtr ;

		resultPtr = &resultImg ;
		mergePtr = &mergeImg ;

		cout << "Please enter the path of image." << endl ;

		//Check the entered file path is available or not.
		do
		{
			//Alert user that the entered path is wrong if loading is not correctly.
			if( errInput )
			{
				cout << "Wrong file path! Please enter again." << endl ;
			}
			cout << "Path:" ;
			cin >> imgPath ;

			//Load the image from input path.
			image = imread( imgPath ) ;

			//If the loading is successful, errInput will be set as false.
			errInput = !( image ).data ;

		}while( errInput ) ;

		cout << "Which kind of face mosaics method you want to choose?" << endl ;
		cout << "1)Blur 2)Dilate 3)Gaussian Blur 4)Median Blur 5)Eyes Mask :" ;

		do
		{
			if( errInput )
			{
				cout << "Input error, please try again." << endl ;
				cout << "1)Blur 2)Dilate 3)Gaussian Blur 4)Median Blur 5)Eyes Mask :" ;
			}
			cin >> methods ;
			if( methods > 0 && methods < 6 )
			{
				errInput = false ;
			}
			else
			{
				errInput = true ;
			}

		}while( errInput ) ;

		resultPtr = faceDetection( &image , resultPtr , methods ) ;
		mergePtr = beforeAfterMerge( &image , resultPtr , mergePtr ) ;

		imshow( "Before and After" , mergeImg ) ;

		cout << "Please press any key to continue." << endl ;
		waitKey( 0 ) ;

		//Check if user want to save the processed image or not.
		do
		{
			cout << "Save the processed image?(Y/N):" ;
			cin >> input ;
			switch( input )
			{
				//If yes, then save the processed image.
				case 'y' :
				case 'Y' :
					fileName = itoa( saved + 1 , buffer , 10 ) ;
					fileName += ".jpg" ;
					imwrite( fileName , mergeImg ) ;
					saved ++ ;
					errInput = false ;
					break ;

				//If no, then break the checking.
				case 'n' :
				case 'N' :
					errInput = false ;
					break ;

				//Input is neigher yes nor no, let user retry again.
				default :
					errInput = true ;
					break ;
			}
		}while( errInput ) ;

		//Check if user want to process another image or not.
		do
		{
			cout << "Process another image?(Y/N):" ;
			cin >> input ;
			errInput = false ;
			switch( input )
			{
				//If yes, then continue.
				case 'y' :
				case 'Y' :
					conti = true ;
					break ;

				//If no, then terminate the program.
				case 'n' :
				case 'N' :
					conti = false ;
					break ;

				//Input is neither yes nor no, let user retry again.
				default :
					errInput = true ;
					break ;
			}
		}while( errInput ) ;

		//Clean up the screen.
		system( "cls" ) ;
	}while( conti ) ;

	cout << "Program was terminated." << endl ;

	system( "pause" ) ;
	return 0 ;
}

//Face detection function.
Mat *faceDetection( Mat *orig , Mat *result , int methods )
{
	( *orig ).copyTo( *result ) ;
	Mat grayImage , grayFaceROI , colorFaceROI ;

	//Convert the original image into gray scale image.
	cvtColor( *result , grayImage , CV_BGR2GRAY ) ;

	//Equalizes the histogram of a grayscale image.
	equalizeHist( grayImage, grayImage );

	//Define the variable of classifier and load face classifier.
	CascadeClassifier cascadeFace ;
	cascadeFace.load( faceFilePath ) ;

	//Define a vector of face in rectangle type.
	vector< Rect > faceVector ;

	//Detect face in the gray image of original image.
	cascadeFace.detectMultiScale( grayImage , faceVector , 1.1 , 2 , 0 , Size( 5 ,5 ) , Size( 500 , 500 ) ) ;

	switch( methods )
	{
		//Blur.
		case 1 :
			for( int i = 0 ; i < faceVector.size( ) ; i ++ )
			{
				//Get the vector of face and set it as a ROI image with color.
				colorFaceROI = ( *result )( faceVector[ i ] ) ;

				//Call face blur function to blurs the part of face.
				faceBlur( &colorFaceROI ) ;
			}
			break ;

		//Dilate.
		case 2 :
			for( int i = 0 ; i < faceVector.size( ) ; i ++ )
			{
				//Get the vector of face and set it as a ROI image with color.
				colorFaceROI = ( *result )( faceVector[ i ] ) ;

				//Call face dilation function to dilates the part of face.
				faceDilation( &colorFaceROI ) ;
			}
			break ;

		//Gaussian blur.
		case 3 :
			for( int i = 0 ; i < faceVector.size( ) ; i ++ )
			{
				//Get the vector of face and set it as a ROI image with color.
				colorFaceROI = ( *result )( faceVector[ i ] ) ;

				//Call face blur function to blurs the part of face.
				faceGaussianBlur( &colorFaceROI ) ;
			}
			break ;

		//Median blur.
		case 4 :
			for( int i = 0 ; i < faceVector.size( ) ; i ++ )
			{
				//Get the vector of face and set it as a ROI image with color.
				colorFaceROI = ( *result )( faceVector[ i ] ) ;

				//Call face blur function to blurs the part of face.
				faceMedianBlur( &colorFaceROI ) ;
			}
			break ;

		//Eyes mask
		case 5 :
			for( int i = 0 ; i < faceVector.size( ) ; i ++ )
			{
				//Get the vector of face and set it as a ROI image in gray scale.
				grayFaceROI = grayImage( faceVector[ i ] ) ;

				//Call eye detection sub-function to find eyes position.
				result = eyesDetection( result , &grayFaceROI , &faceVector[ i ] ) ;
			}
			break ;
	}

	return result ;
}

//Eyes detection function.
Mat *eyesDetection( Mat *img , Mat *roi ,   Rect *faceRegion )
{
	//Define the variable of classifier and load eyes classifier.
	CascadeClassifier cascadeEyes ;
	cascadeEyes.load( eyesFilePath ) ;

	//Define a vector of eyes in rectangle type.
	vector< Rect > eyesVector ;

	//Detect eyes in the ROI area of face vector.
	cascadeEyes.detectMultiScale( *roi , eyesVector , 1.1 , 1 , 0 , Size( 0.5 , 0.5 ) ) ;

	int uprY = INT_MAX , lwrY = 0 ;
	Point upperLeft , lowerRight ;

	//Compute the range of eyes mask.
	for( int j = 0 ; j < eyesVector.size( ) ; j ++ )
	{
		if( eyesVector[ j ].y < uprY )
		{
			uprY = eyesVector[ j ].y ;
		}
		if( eyesVector[ j ].y + eyesVector[ j ].height > lwrY )
		{
			lwrY = eyesVector[ j ].y + eyesVector[ j ].height ;
		}
	}

	//Set the range of eyes mask.
	if( eyesVector.size( ) > 0 )
	{
		upperLeft.x = ( *faceRegion ).x ;
		upperLeft.y = ( *faceRegion ).y + uprY ;
		lowerRight.x = ( *faceRegion ).x + ( *faceRegion ).width ;
		lowerRight.y = ( *faceRegion ).y + lwrY ;
	}

	//Draw the eyes mask on face.
	rectangle( *img , upperLeft , lowerRight , Scalar( 0 , 0 , 0 ) , CV_FILLED ) ;

	return img ;
}

void faceBlur( Mat *colorFaceROI )
{
	Mat *bluredFace = colorFaceROI ;

	//Do face blur. If the Size is more bigger, the effect of blur will be better.
	blur( *colorFaceROI , *bluredFace , Size( 21 , 21 ) , Point(-1, -1 ) , BORDER_DEFAULT ) ;

	( *bluredFace ).copyTo( *colorFaceROI ) ;
}

void faceDilation( Mat *colorFaceROI )
{
	Mat *dilatedFace = colorFaceROI ;
	Mat element ;

	//Set the structuring element used for dilation.
	element = getStructuringElement( MORPH_RECT , Size( 7 , 7 ) , Point( -1  ,-1 ) ) ;

	//Do face dilation.
	dilate( *colorFaceROI , *dilatedFace , element , Point(-1, -1 ) , 1 ) ;

	( *dilatedFace ).copyTo( *colorFaceROI ) ;
}

void faceGaussianBlur( Mat *colorFaceROI )
{
	Mat *gaussianBluredFace = colorFaceROI ;

	//Do face gaussian blur. If the Size is more bigger, the effect of blur will be better.
	GaussianBlur( *colorFaceROI , *gaussianBluredFace , Size( 21 , 21 ) , 0 , 0 , BORDER_DEFAULT ) ;

	( *gaussianBluredFace ).copyTo( *colorFaceROI ) ;
}

void faceMedianBlur( Mat *colorFaceROI )
{
	Mat *medianBluredFace = colorFaceROI ;

	//Do face gaussian blur. If the 3rd parameter is more bigger, the effect of blur will be better.
	medianBlur( *colorFaceROI , *medianBluredFace , 21 ) ;

	( *medianBluredFace ).copyTo( *colorFaceROI ) ;
}

Mat *beforeAfterMerge( Mat *original , Mat *processed , Mat *merge )
{
	//Create a image that 2 times width.
	( *merge ).create( ( *original ).rows , ( *original ).cols * 2 , ( *original ).type( ) ) ;

	//Merge the original image and processed image together.
	( *original ).copyTo( ( *merge )( Rect( 0 , 0 , ( *original ).cols , ( *original ).rows ) ) ) ;
	( *processed ).copyTo( ( *merge )( Rect( ( *original ).cols , 0 , ( *processed ).cols , ( *processed ).rows ) ) ) ;

	return merge ;
}
