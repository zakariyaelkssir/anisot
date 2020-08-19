/*
 * Copyright (c) 2013, Marc Lebrun <marc.lebrun.ik@gmail.com>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include "Utilities/Utilities.h"
#include "NlBayes/NlBayes.h"
#include "Utilities/LibImages.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
using namespace std;



/**
 * @file   main.cpp
 * @brief  Main executable file
 *
 *
 *
 * @author MARC LEBRUN  <marc.lebrun.ik@gmail.com>
 **/

int main(int argc, char **argv)
{
    //! Check if there is the right call for the algorithm
	if (argc < 13) {
		cout << "usage: NL_Bayes image sigma noisy denoised basic difference \
		bias basic_bias diff_bias useArea1 useArea2 computeBias" << endl;
		return EXIT_FAILURE;
	}

    //! Variables initialization
	const float sigma   = atof(argv[2]);
	const bool doBias   = (bool) atof(argv[12]);
	const bool useArea1 = (bool) atof(argv[10]);
	const bool useArea2 = (bool) atof(argv[11]);
	const bool verbose  = true;


	//! Declarations
	vector<float> im, imNoisy, im_aniso, imFinal, imDiff;
	vector<float> imBias, imBasicBias, imDiffBias;
	ImageSize imSize,imSize_sym;

    //! Load image
	if(loadImage(argv[1], im, imSize, verbose) != EXIT_SUCCESS) {
        return EXIT_FAILURE;
	}
	//im_aniso=im;
	// anisotropicSmoothing
	//for(unsigned i=0;i<20;i++)
	if(anisotropicSmoothing(im,imSize)!= EXIT_SUCCESS){
		return EXIT_FAILURE;
	}

	if (saveImage("im_aniso.png", im, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}
	return 0;
	// normalize image  
	/*if(normalize_image(im,im_normalize_image,imSize,0,1)!= EXIT_SUCCESS){
		return EXIT_FAILURE;
	}*/
	vector<float> grady,gradx;
	/*if(gradientX(im,gradx,imSize,1)!= EXIT_SUCCESS){
		return EXIT_FAILURE;
	}*/

	
	//Variable 

	float myalpha = 0.001,sig=0.5;
	unsigned T = 15,rho = 0.2;
	int C= 1;
	float stepT=10000.15;
	// %% 1 gaussian K_sigma
	float t=0.f,s=0;

	int limitXSize = ceil(2*sig)-(-ceil(2*sig))+1;
	vector<float>limitX(limitXSize),kSigma(limitXSize)	;
	for(int i = -ceil(2*sig),j=0; i < ceil(2*sig)+1; i++,j++){
    limitX[j] = i;
	kSigma[j] = exp(-(i*i)/(2*pow(sig,2)));
	s+=kSigma[j];
	}
	for(int i = -ceil(2*sig),j=0; i < ceil(2*sig)+1; i++,j++){
	kSigma[j]=kSigma[j]/s;
		}	
		
	vector<float>output,img_sym;
	float *mask   = new float[limitXSize*limitXSize]; 
	int c = 0;
	for(int i = 0;i<limitXSize;i++){
		for(int j =0;j<limitXSize;j++){
			mask[i*limitXSize+j]=kSigma[i]*kSigma[j];
			
					}
	}
	// generate mask
	int limitXSizeJ = ceil(3*rho)-(-ceil(3*rho))+1;
	vector<float>kSigmaJ(limitXSizeJ)	;
	s=0;
	for(int i = -ceil(3*rho),j=0; i < ceil(3*rho)+1; i++,j++){
	kSigmaJ[j] = exp(-(i*i)/(2*pow(rho,2)));
	s+=kSigmaJ[j];
	}
	for(int i = -ceil(3*rho),j=0; i < ceil(3*rho)+1; i++,j++){
	kSigmaJ[j]=kSigmaJ[j]/s;
		}	
	float *mask2   = new float[limitXSizeJ*limitXSizeJ]; 
	for(int i = 0;i<limitXSizeJ;i++){
		for(int j =0;j<limitXSizeJ;j++){
			mask2[i*limitXSizeJ+j]=kSigmaJ[i]*kSigmaJ[j];
					}
	}
	output.resize(im.size());
	Filter2d(im,output,imSize,mask,limitXSize/2,limitXSize);
	//calcule gradient 
/*	for(unsigned i =imSize.width*2;i<imSize.width*3;i++){
		cout<<output[i]<<endl;
	}*/
	 if (saveImage("8_sym.png", output, imSize, 0.f, 255.f) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}  
	cout<<output.size();
	return 0;
   // symetrize(im,img_sym,w_b,h_b,imSize.nChannels,2);

	gradientY(output,grady,imSize,1);
    gradientX(output,gradx,imSize,1);

	//calcule Jxx , Jyy, Jxu 

	vector<float> Jxx,Jxy,Jyy,Jyy_n,Jxy_n,Jxx_n;
	Jxx.resize(im.size());
	Jxy.resize(im.size());
	Jyy.resize(im.size());
	Jyy_n.reserve(im.size());
	Jxy_n.reserve(im.size());
	Jxx_n.reserve(im.size());
	for(unsigned i = 0;i<imSize.width;i++){
		for(unsigned j =0;j<imSize.height;j++){
			Jyy[i*imSize.width+j]=grady[i*imSize.width+j]*grady[i*imSize.width+j];
			Jxx[i*imSize.width+j]=gradx[i*imSize.width+j]*gradx[i*imSize.width+j];
			Jxy[i*imSize.width+j]=grady[i*imSize.width+j]*gradx[i*imSize.width+j];
					}
	}

	Filter2d(Jyy,Jyy_n,imSize,mask,limitXSizeJ/2,limitXSizeJ);
	Filter2d(Jxy,Jxy_n,imSize,mask,limitXSizeJ/2,limitXSizeJ);
	Filter2d(Jxx,Jxx_n,imSize,mask2,limitXSizeJ/2,limitXSizeJ);
	std::cout<<limitXSizeJ;
	

	vector<float> v2x,v2y,evec,eval,lamda1,lamda2,v1x,v1y,di,Dxx,Dyy,Dxy,pixel;
	v2x.resize(im.size());
	v2y.resize(im.size());
	lamda1.resize(im.size());
	lamda2.resize(im.size());
	di.resize(im.size());
	Dxx.resize(im.size());
	Dyy.resize(im.size());
	Dxy.resize(im.size());
	pixel.resize(4);
	Eigen::Matrix2d A;
	Eigen::Matrix<float, 2, 2> B;
	//A.resize(4);
	for(unsigned i =0 ; i<imSize.height;i++){
		for(unsigned j=0;j<imSize.width;j++){
			A(0,0)=Jyy[i*imSize.width+j];
			A(0,1)=Jxy[i*imSize.width+j];
			A(1,0)=Jxy[i*imSize.width+j];
			A(1,1)=Jyy[i*imSize.width+j];
			//Eigen::EigenSolver<Eigen::Matrix<float, 2,2> > s(B);
			 Eigen::EigenSolver<Eigen::Matrix2d> es(A);
			 Eigen::Matrix2d D = es.pseudoEigenvalueMatrix();
			 Eigen::Matrix2d V = es.pseudoEigenvectors();
			 v2x[i*imSize.width+j]=V(0,1);
			 v2y[i*imSize.width+j]=V(1,1);
			 lamda1[i*imSize.width+j]=D(1,1);
			 lamda2[i*imSize.width+j]=D(0,0);
			 v2x[i*imSize.width+j]/=-sqrt(pow(v2x[i*imSize.width+j],2)+pow(v2y[i*imSize.width+j],2));
			 v2y[i*imSize.width+j]/=-sqrt(pow(v2x[i*imSize.width+j],2)+pow(v2y[i*imSize.width+j],2));
			 di[i*imSize.width+j]=lamda1[i*imSize.width+j]-lamda2[i*imSize.width+j];
			//return 0;
		}
	}

	for(unsigned i =0 ; i<imSize.height;i++)
		for(unsigned j=0;j<imSize.width;j++){
			lamda1[i*imSize.width+j]=myalpha +(1-myalpha)*exp(-1/(pow(di[i*imSize.width+j],2)));
			Dxx[i*imSize.width+j]=(pow(v2y[i*imSize.width+j],2)*lamda1[i*imSize.width+j])+(pow(v2x[i*imSize.width+j],2)*myalpha);
			Dyy[i*imSize.width+j]=(pow(v2x[i*imSize.width+j],2)*lamda1[i*imSize.width+j])+(pow(v2y[i*imSize.width+j],2)*myalpha);
			Dxy[i*imSize.width+j]=((v2x[i*imSize.width+j]*v2y[i*imSize.width+j])*lamda1[i*imSize.width+j])+(pow(v2x[i*imSize.width+j],2)*myalpha);
		}
	

	//orientationImage
	/*if(ridgeOrient(im_normalize_image, gradientsigma, blocksigma,
										 orientsmoothsigma)!= EXIT_SUCCESS){
		return EXIT_FAILURE;
	}*/

	//calcul gradien 

    //! save noisy, denoised and differences images
	if (verbose) {
	    cout << "Save images...";
	}

	if (saveImage("8_sym.png", imDiff, imSize_sym, 0.f, 255.f) != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}
    if (verbose) {
        cout << "done." << endl;
    }


	#define smooth_kernel_size 7
	#define sigma 1.0
	#define K  1

	
	








	return EXIT_SUCCESS;
}
