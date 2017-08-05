// USEQsuperpixel_TIP.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "USEQsuperpixel_TIP.h"


using namespace cv;
int _tmain(int argc, _TCHAR* argv[])
{

	cv::Mat m_input =imread("testIMG.jpg"); //test image
	cv::Mat m_label,m_colorLabel,m_colorLabelandContour,m_contour = m_input.clone();
	cv::medianBlur ( m_input, m_input, 3 );	
	USEQsuperpixel_TIP sp;	
	    //paremeter set if need
    	//sp.localDominateMag = 0.2f ; 
		//sp.refinementMag = 0.2f ;
		//sp.setQtzLv = 3;
	
	sp.Cluster(m_input,m_label,100,0.05f );	//do USEQ  (inputImg,outputLabel,superpixel numbers,omega)
	sp.Label2Color(m_label,m_colorLabel);   //draw color label
	sp.LabelContourMask(m_label,m_contour); //draw contour
	m_colorLabelandContour = 0.5*m_input + 0.5*m_colorLabel;  //draw translucent map
	sp.LabelContourMask(m_label,m_colorLabelandContour);      //draw translucent map contour

		
	cv::imshow("m_input",m_input);
	cv::imshow("m_colorLabel",m_colorLabel);
	cv::imshow("m_contour",m_contour);
	cv::imshow("m_colorLabelandContour",m_colorLabelandContour);
	waitKey(0);




	return 0;
}

