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

/**
 * @file NlBayes.cpp
 * @brief NL-Bayes denoising functions
 *
 * @author Marc Lebrun <marc.lebrun.ik@gmail.com>
 **/

#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <math.h>

#include "NlBayes.h"
#include "LibMatrix.h"
#include "../Utilities/LibImages.h"
#include "../Utilities/Utilities.h"
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

/**
 * @brief Initialize Parameters of the NL-Bayes algorithm.
 *
 * @param o_paramStep1 : will contain the nlbParams for the first step of the algorithm;
 * @param o_paramStep2 : will contain the nlbParams for the second step of the algorithm;
 * @param p_sigma : standard deviation of the noise;
 * @param p_imSize: size of the image;
 * @param p_useArea1 : if true, use the homogeneous area trick for the first step;
 * @param p_useArea2 : if true, use the homogeneous area trick for the second step;
 * @param p_verbose : if true, print some informations.
 *
 * @return none.
 **/


// calcule mean 

float mean(
	std::vector<float> &Image
,	const ImageSize &p_imSize
){
	float mean = 0.f;
	for(unsigned i =0;i<p_imSize.wh;i++){
		mean+=Image[i];
	}
	mean/=((float)(p_imSize.height*p_imSize.width));
	return mean;
}


float deviation(
	std::vector<float> &o_imBasic
,	const ImageSize &p_imSize
,	const unsigned mean
){
	float sdev = 0.0;
	for (unsigned i = 0; i < p_imSize.height; i++) {
        for (unsigned j = 0; j < p_imSize.width; j++) {
            float pixel = o_imBasic[i*p_imSize.width+j];
            float dev = (pixel - mean) * (pixel - mean);
            sdev = sdev + dev;
        }
    }

    int size = p_imSize.width * p_imSize.height;
    float var = sdev / (size - 1);
    float sd = sqrt(var);

    return sd;
}


int normalize_image(
	std::vector<float> const& i_imNoisy
,   std::vector<float> &normalizedImage
,	const ImageSize &p_imSize
,   const unsigned reqMean
, 	const unsigned reqVar
){
	vector<float> imNoisy = i_imNoisy;
	normalizedImage.resize(i_imNoisy.size());
	// calcule mean 
	float mean_image=mean(imNoisy,p_imSize);

	for(unsigned i =0;i<p_imSize.whc;i++){
		normalizedImage[i]=imNoisy[i]-mean_image;
	}
	// calcule mean 
    mean_image = mean(normalizedImage,p_imSize);
	float stdNorm = deviation(normalizedImage,p_imSize,mean_image);
	cout<<stdNorm<<std::endl;
	//stdNorm=75.9435;
	for(unsigned i =0;i<p_imSize.wh;i++){
		normalizedImage[i]=normalizedImage[i]/stdNorm;
		normalizedImage[i] = reqMean + normalizedImage[i]*sqrt(reqVar);
	}
return EXIT_SUCCESS;

}



int gradientY(
	std::vector<float> const& image
,   std::vector<float> &Grady
,	const ImageSize &p_imSize
,   const unsigned spacing
){
	std::vector<float> centeredMat,offsetMat,resultCenteredMat;
	//initialisation with 0 
	Grady.resize(p_imSize.whc);
	for(unsigned i = 0 ; i<p_imSize.whc;i++) Grady[i]=0;
	/*  last row */

	/* get gradients in each border */
    /* first row */
	for(unsigned i = 0 ; i<p_imSize.width;i++) {
			Grady[i]=-image[i]+image[i+p_imSize.width];
	}
    /* last row */
	for(unsigned i = (p_imSize.height-2)*p_imSize.width ; i<(p_imSize.height-1)*p_imSize.width;i++) {
			Grady[i]=-image[i]+image[i+p_imSize.width];
	}
	//
	centeredMat.resize(p_imSize.width*(p_imSize.height-2));
	offsetMat.resize(p_imSize.width*(p_imSize.height-2));
	resultCenteredMat.resize(p_imSize.width*(p_imSize.height-2));
	for(unsigned i = 0 ; i<p_imSize.width-2;i++) 
	for(unsigned j = 0 ; j<p_imSize.height;j++) {
		centeredMat[i*p_imSize.width+j]=image[i*p_imSize.width+j];
		offsetMat[i*p_imSize.width+j]=image[i*p_imSize.width+j+p_imSize.width*2];
		resultCenteredMat[i*p_imSize.width+j]=(offsetMat[i*p_imSize.width+j]-centeredMat[i*p_imSize.width+j])/(((float)spacing)*2.0);
	}

	for(unsigned i = 0 ; i<p_imSize.width*(p_imSize.height-2);i++) {
		Grady[i+p_imSize.width]=resultCenteredMat[i];
	}

return EXIT_SUCCESS;

}

int gradientX(
	std::vector<float> const& image
,   std::vector<float> &Gradx
,	const ImageSize &p_imSize
,   const unsigned spacing
){
	std::vector<float> centeredMat,offsetMat,resultCenteredMat;
	//initialisation with 0 
	Gradx.resize(p_imSize.whc);
	for(unsigned i = 0 ; i<p_imSize.whc;i++) Gradx[i]=0;
	/*  last row */
	/* get gradients in each border */
    /* first col */
	for(unsigned i = 0 ; i<p_imSize.height;i++) {
			Gradx[i*p_imSize.width]=-image[i*p_imSize.width]+image[i*p_imSize.width+1];
			
	}
	
    /* last col */
	for(unsigned i = 0 ; i<p_imSize.height;i++) {
	Gradx[i*p_imSize.width+p_imSize.width-1]=-image[i*p_imSize.width+(p_imSize.width-2)]+image[i*p_imSize.width+(p_imSize.width-1)];
	}
	
	//
	centeredMat.resize((p_imSize.width-2)*p_imSize.height);
	offsetMat.resize((p_imSize.width-2)*p_imSize.height);
	resultCenteredMat.resize((p_imSize.width-2)*p_imSize.height);
	for(unsigned i = 0 ; i<p_imSize.width-2;i++) 
	for(unsigned j = 0 ; j<p_imSize.height;j++) {
		centeredMat[j*(p_imSize.width-2)+i]= image[j*(p_imSize.width)+i];
		offsetMat[j*(p_imSize.width-2)+i]=image[j*(p_imSize.width)+i+2];
		resultCenteredMat[j*(p_imSize.width-2)+i]=(offsetMat[j*(p_imSize.width-2)+i]-centeredMat[j*(p_imSize.width-2)+i])/(((float)spacing)*2.0);
	}
	
	for(unsigned i = 0 ; i<p_imSize.width-2;i++) 
	for(unsigned j = 0 ; j<p_imSize.height;j++) {
		Gradx[j*p_imSize.width+i+1]=resultCenteredMat[j*(p_imSize.width-2)+i];
	}
return EXIT_SUCCESS;

}

// gaussien Kernel
void getGaussianKernel(
	std::vector<float> &gauss
,	unsigned smooth_kernel_size
, 	double sigma
,	int K){
    double sum = 0;
    unsigned i, j;
	gauss.resize(smooth_kernel_size);
    for (i = 0; i < smooth_kernel_size; i++) {
            double x = i - (smooth_kernel_size - 1) / 2.0;
            gauss[i]= K * exp(((pow(x, 2) / ((2 * pow(sigma, 2)))) * (-1)));
            sum += gauss[i];
    }
    for (i = 0; i < smooth_kernel_size; i++) {
            gauss[i] /= sum;
    }

}

int ridgeOrient(
	std::vector<float> const& im_normalize_image
,	const double gradientsigma
,	const double blocksigma
,	const double orientsmoothsigma

){
	int sze = 6 * round(gradientsigma);

	if (sze % 2 == 0) {
		sze++;
	}
	//// Define Gaussian kernel
	std::vector<float> gaussKernelX;
	std::vector<float> gaussKernelY;
	std::vector<float> gaussKernel;
	getGaussianKernel(gaussKernelX,sze,gradientsigma,1);
	getGaussianKernel(gaussKernelY,sze,gradientsigma,1);

	    for (int i = 0; i < sze; i++) {
            printf("%f ", gaussKernelX[i]);
	}

	vector<float>kernelx(3);
	kernelx[0]=0;

	return EXIT_SUCCESS;

}



int Filter2d(
	std::vector<float> const& input
,   std::vector<float> &output
,	const ImageSize &p_imSize
,   std::vector<float> mask
,   unsigned band
, 	unsigned tailemask){
	unsigned ny = p_imSize.height;
	unsigned nx = p_imSize.width;
    //Symetrise
	vector<float> out;
	const unsigned h_b = p_imSize.height + 2 * band;
    const unsigned w_b = p_imSize.width  + 2 * band;
	symetrizeImage(input,out,p_imSize,band,true);
	//apply the mask to the center body of the image
    for(unsigned i = band,i1=0; i < h_b-band; i++,i1++)
    {
	for(unsigned j = band,j1=0; j < w_b-band; j++,j1++)
	{
	    double sum = 0;
	    for(int l = 0; l < tailemask; l++)
	    {
		for(unsigned m = 0; m < tailemask; m++)
		{
		    unsigned p = (i + l -band) * h_b + j + m -band;
		    sum += out[p] * mask[l * tailemask + m];
		}
	    }
	    unsigned k = i1 * nx + j1;
	    output[k] = sum;
	}
    }
	/*
    //apply the mabfsk to the first and last rows
    for(unsigned j = 1; j < nx-1; j++)
    {
	double sum = 0;
	sum += input[j-1] * (mask[0] + mask[3]);
	sum += input[ j ] * (mask[1] + mask[4]);
	sum += input[j+1] * (mask[2] + mask[5]);

	sum += input[nx + j-1] * mask[6];
	sum += input[nx +  j ] * mask[7];
	sum += input[nx + j+1] * mask[8];

	output[j] = sum;

	sum = 0;
	sum += input[(ny-2)*nx+j-1] * mask[0];
	sum += input[(ny-2)*nx+j  ] * mask[1];
	sum += input[(ny-2)*nx+j+1] * mask[2];

	sum += input[(ny-1)*nx+j-1] * (mask[6] + mask[3]);
	sum += input[(ny-1)*nx+j  ] * (mask[7] + mask[4]);
	sum += input[(ny-1)*nx+j+1] * (mask[8] + mask[5]);

	output[(ny-1)*nx + j] = sum;
    }

    //apply the mask to the first and last columns
    for(unsigned i = 1; i < ny-1; i++)
    {
	double sum = 0;
	sum += input[(i - 1)*nx]   * (mask[0] + mask[1]);
	sum += input[(i - 1)*nx+1] * mask[2];

	sum += input[i * nx]   * (mask[3] + mask[4]);
	sum += input[i * nx+1] * mask[5];

	sum += input[(i + 1)*nx]   * (mask[6] + mask[7]);
	sum += input[(i + 1)*nx+1] * mask[8];

	output[i*nx] = sum;

	sum = 0;
	sum += input[i * nx-2] * mask[0];
	sum += input[i * nx-1] * (mask[1] + mask[2]);

	sum += input[(i + 1)*nx-2] * mask[3];
	sum += input[(i + 1)*nx-1] * (mask[4] + mask[5]);

	sum += input[(i + 2)*nx-2] * mask[6];
	sum += input[(i + 2)*nx-1] * (mask[7] + mask[8]);

	output[i*nx + nx -1] = sum;
    }

    //apply the mask to the four corners
    output[0] = input[0]    * (mask[0] + mask[1] + mask[3] + mask[4]) +
	input[1]    * (mask[2] + mask[5]) +
	input[nx]   * (mask[6] + mask[7]) +
	input[nx+1] * mask[8];

    output[nx-1] =
	input[nx-2]   * (mask[0] + mask[3]) +
	input[nx-1]   * (mask[1] + mask[2] + mask[4] + mask[5]) +
	input[2*nx-2] * mask[6] +
	input[2*nx-1] * (mask[7] + mask[8]);

    output[(ny-1)*nx] =
	input[(ny-2)*nx]   * (mask[0] + mask[1]) +
	input[(ny-2)*nx+1] *  mask[2] +
	input[(ny-1)*nx]   * (mask[3] + mask[4] + mask[6] + mask[7]) +
	input[(ny-1)*nx+1] * (mask[5] + mask[8]);

    output[ny*nx-1] =
	input[(ny-1)*nx-2] * mask[0] +
	input[(ny-1)*nx-1] * (mask[1] + mask[2]) +
	input[ny*nx-2] * (mask[3] + mask[6]) +
	input[ny*nx-1] * (mask[4] + mask[5] + mask[7] + mask[8]);
	*/
    }



void NonNegativityDiscretization(
    std::vector<float> const& input
,   std::vector<float> &output
,	const ImageSize &p_imSize
,   std::vector<float> & Dxx
,   std::vector<float> & Dyy
, 	std::vector<float> & Dxy
,	const float Step
){
	output.resize(input.size());
	vector<float> px,py,ny,nx,wbR1,wbL3,wtR7,wtL9,wtM2,wmR4,wmB8,wmL6,image_filter;
	wbR1.resize(input.size());
	wbL3.resize(input.size());
	wtR7.resize(input.size());
	wtL9.resize(input.size());
	wtM2.resize(input.size());
	wmR4.resize(input.size());
	wmL6.resize(input.size());
	wmB8.resize(input.size());
	px.resize(p_imSize.width);
	py.resize(p_imSize.height);
	ny.resize(p_imSize.width);
	nx.resize(p_imSize.height);
	nx[0]=0;
	ny[0]=0;
	for(unsigned i=0 ;i<p_imSize.width ;i++){
		px[i]=i+1;
		ny[i+1]=i;
	}
	for(unsigned i=0 ;i<p_imSize.height ;i++){
		py[i]=i+1;
		nx[i+1]=i;
	}
	px[p_imSize.width-1]=255;
	py[p_imSize.height-1]=255;
	// % Stencil Weights
	vector<float> c,c2;
	c.resize(input.size());
	c2.resize(input.size());

	for(unsigned i =0;i<p_imSize.whc;i++){
		c[i]=abs(Dxy[i])-Dxy[i];
		c2[i]=abs(Dxy[i])+Dxy[i];
		
	}

	for(unsigned i =0;i<p_imSize.height;i++){
		for(unsigned j =0;j<p_imSize.width;j++){
		wbR1[i*p_imSize.width+j]=0.25*(abs(Dxy[nx[i]*p_imSize.width+py[j]])-Dxy[nx[i]*p_imSize.width+py[j]]+c[i*p_imSize.width+j]);
		wbL3[i*p_imSize.width+j]=0.25*(abs(Dxy[px[i]*p_imSize.width+py[j]])+Dxy[px[i]*p_imSize.width+py[j]]+c2[i*p_imSize.width+j]);
		wtR7[i*p_imSize.width+j]=0.25*(abs(Dxy[nx[i]*p_imSize.width+py[j]])+Dxy[nx[i]*p_imSize.width+py[j]]+c2[i*p_imSize.width+j]);
		wtL9[i*p_imSize.width+j]=0.25*(abs(Dxy[px[i]*p_imSize.width+py[j]])-Dxy[px[i]*p_imSize.width+py[j]]+c[i*p_imSize.width+j]);

		//wtM2.at<float>(i,j) = 0.5*((Dyy.at<float>(i,py.at<float>(0,j))+Dyy.at<float>(i,j))-(abs(Dxy.at<float>(i,py.at<float>(0,j)))+abs(Dxy.at<float>(i,j))));


		//wtM2[i*p_imSize.width+j]=0.5*(Dyy[i*p_imSize.width+py[j]]+Dyy[i*p_imSize.width+j]-abs(Dxy[i*p_imSize.width+py[j]])+abs(Dxy[i*p_imSize.width+j]));
		wtM2[i*p_imSize.width+j]=0.5*((Dyy[i*p_imSize.width+py[j]]+Dyy[i*p_imSize.width+j])-(abs(Dxy[i*p_imSize.width+py[j]])+abs(Dxy[i*p_imSize.width+j])));

		wmR4[i*p_imSize.width+j]=0.5*(Dxx[nx[i]*p_imSize.width+j]+Dxx[i*p_imSize.width+j]-(abs(Dxy[nx[i]*p_imSize.width+j])+abs(Dxy[i*p_imSize.width+j])));
		wmL6[i*p_imSize.width+j]=0.5*(Dxx[px[i]*p_imSize.width+j]+Dxx[i*p_imSize.width+j]-(abs(Dxy[px[i]*p_imSize.width+j])+abs(Dxy[i*p_imSize.width+j])));
		wmB8[i*p_imSize.width+j]=0.5*(Dyy[i*p_imSize.width+ny[j]]+Dyy[i*p_imSize.width+j]-(abs(Dxy[i*p_imSize.width+ny[j]])+abs(Dxy[i*p_imSize.width+j])));

				
		//image_filter.at<float>(i,j)=image.at<float>(i,j)+step*((wbR1.at<float>(i,j)*(image.at<float>(nx.at<float>(0,i),py.at<float>(0,j))-image.at<float>(i,j))));//+(wtM2.at<float>(i,j)*(image.at<float>(i,py.at<float>(0,j))-image.at<float>(i,j)))+(wbL3.at<float>(i,j)*(image.at<float>(px.at<float>(0,i),py.at<float>(0,j))-image.at<float>(i,j)))+(wmR4.at<float>(i,j)*(image.at<float>(nx.at<float>(0,i),j)-image.at<float>(i,j)))+(wmL6.at<float>(i,j)*(image.at<float>(px.at<float>(0,i),j)-image.at<float>(i,j)))+(wtR7.at<float>(i,j)*(image.at<float>(nx.at<float>(0,i),ny.at<float>(0,j))-image.at<float>(i,j)))+(wmB8.at<float>(i,j)*(image.at<float>(i,ny.at<float>(0,j))-image.at<float>(i,j)))+(wtL9.at<float>(i,j)*(image.at<float>(px.at<float>(0,i),ny.at<float>(0,j))-image.at<float>(i,j))));
		
		output[i*p_imSize.width+j]=	input[i*p_imSize.width+j]+Step*((wbR1[i*p_imSize.width+j]*(input[nx[i]*p_imSize.width+py[j]]-input[i*p_imSize.width+j]))+(wbL3[i*p_imSize.width+j]*(input[px[i]*p_imSize.width+py[j]]-input[i*p_imSize.width+j]))+(wmR4[i*p_imSize.width+j]*(input[nx[i]*p_imSize.width+j]-input[i*p_imSize.width+j]))+(wmL6[i*p_imSize.width+j]*(input[px[i]*p_imSize.width+j]-input[i*p_imSize.width+j]))+(wtR7[i*p_imSize.width+j]*(input[nx[i]*p_imSize.width+ny[j]]-input[i*p_imSize.width+j]))+(wmB8[i*p_imSize.width+j]*(input[i*p_imSize.width+ny[j]]-input[i*p_imSize.width+j]))+(wtL9[i*p_imSize.width+j]*(input[px[i]*p_imSize.width+ny[j]]-input[i*p_imSize.width+j])))+(wtM2[i*p_imSize.width+j]*(input[i*p_imSize.width+py[j]]-input[i*p_imSize.width+j]));	
		

		}
	}
}

void Generate_mask(
	float sig
,	float myalpha
,	std::vector<float> & mask
,	int &limitXSize
,	int coef
){
	float s=0;
	limitXSize = ceil(coef*sig)-(-ceil(coef*sig))+1;
	mask.resize(limitXSize*limitXSize);
	vector<float>kSigma(limitXSize)	;
	
	
	for(int i = -ceil(coef*sig),j=0; i < ceil(coef*sig)+1; i++,j++){
    
	kSigma[j] = exp(-(i*i)/(2*pow(sig,2)));
	s+=kSigma[j];
	}
	for(int i = -ceil(coef*sig),j=0; i < ceil(coef*sig)+1; i++,j++){
	kSigma[j]=kSigma[j]/s;
		}	
	int c = 0;
	for(int i = 0;i<limitXSize;i++){
		for(int j =0;j<limitXSize;j++){
			mask[i*limitXSize+j]=kSigma[i]*kSigma[j];	
		}
		
		
	}
	
}
float g(float l1, float l2)
{
    float C1=0.001, C2=1.0;
    float s=l1-l2;
 if (std::abs(s) <0.0001)
 return C1;
 else
        return    C1+(1-C1)*std::exp(-C2/std::pow(s,2));

}


void spectreTensor2(
		std::vector<float> const& Im
,		std::vector<float> & A
,		std::vector<float> & B
,		std::vector<float> & C
,		const ImageSize &p_imSize

){

		vector<float>dG3;
		dG3.resize(3*3);
			dG3[0]=3.0/32;
			dG3[1]=0;
			dG3[2]=-3.0/32;
			dG3[3]=10.0/32;
			dG3[4]=0;
			dG3[5]=-10.0/32;
			dG3[6]=3.0/32;
			dG3[7]=0;
			dG3[8]=-3.0/32;
		vector<float>output;
		output.resize(Im.size());
		Filter2d(Im,output,p_imSize,dG3,1,3);
		A.resize(Im.size());
		B.resize(Im.size());
		C.resize(Im.size());

		if (saveImage("imdif.png", output, p_imSize, 0.f, 255.f) != EXIT_SUCCESS) {
		return ;
	}
/*

	   float temp1, temp2, L1,L2, C1=0.001,theta, cosTheta, sinTheta, lambda1, lambda2;
	   vector<float> P;
	   P.resize(2);
		for(unsigned i=0;i<p_imSize.height;i++){
			for(unsigned j=0;j<p_imSize.width;j++){
			temp1=Jxx[i*p_imSize.width+j]-Jyy[i*p_imSize.width+j];
            temp2=sqrt(temp1*temp1 +4*Jxy[i*p_imSize.width+j]*Jxy[i*p_imSize.width+j]);
            lambda1=0.5*(temp1+temp2);
            lambda2=0.5*(temp1-temp2);
			P[0]=2*Jxy[i*p_imSize.width+j];
			P[1]=Jyy[i*p_imSize.width+j] - Jxx[i*p_imSize.width+j]+temp2;
			float normp=std::sqrt(P[0]*P[0]+P[1]*P[1]+0.00000001);

             P[0]=(P[0]/normp);
			 P[1]=(P[1]/normp);

		 L1=C1, L2=g(lambda1,lambda2);
         cosTheta=std::cos(theta);
         sinTheta=std::sin(theta);
         A[i*p_imSize.width+j]= L1*P[0]*P[0]+L2*P[1]*P[1];
         B[i*p_imSize.width+j]=(L1-L2)*P[0]*P[1];
         C[i*p_imSize.width+j]= L1*P[1]*P[1]+L2*P[0]*P[0];
			}
		}*/
}




void spectreTensor(
		std::vector<float> const& Jxx
,		std::vector<float> const& Jxy
,		std::vector<float> const& Jyy
,		std::vector<float> & A
,		std::vector<float> & B
,		std::vector<float> & C
,		const ImageSize &p_imSize

){
		A.resize(Jxx.size());
		B.resize(Jxx.size());
		C.resize(Jxx.size());

	   float temp1, temp2, L1,L2, C1=0.001,theta, cosTheta, sinTheta, lambda1, lambda2;
	   vector<float> P;
	   P.resize(2);
		for(unsigned i=0;i<p_imSize.height;i++){
			for(unsigned j=0;j<p_imSize.width;j++){
			temp1=Jxx[i*p_imSize.width+j]-Jyy[i*p_imSize.width+j];
            temp2=sqrt(temp1*temp1 +4*Jxy[i*p_imSize.width+j]*Jxy[i*p_imSize.width+j]);
            lambda1=0.5*(temp1+temp2);
            lambda2=0.5*(temp1-temp2);
			P[0]=2*Jxy[i*p_imSize.width+j];
			P[1]=Jyy[i*p_imSize.width+j] - Jxx[i*p_imSize.width+j]+temp2;
			float normp=std::sqrt(P[0]*P[0]+P[1]*P[1]+0.00000001);

             P[0]=(P[0]/normp);
			 P[1]=(P[1]/normp);

		 L1=C1, L2=g(lambda1,lambda2);
         cosTheta=std::cos(theta);
         sinTheta=std::sin(theta);
         A[i*p_imSize.width+j]= L1*P[0]*P[0]+L2*P[1]*P[1];
         B[i*p_imSize.width+j]=(L1-L2)*P[0]*P[1];
         C[i*p_imSize.width+j]= L1*P[1]*P[1]+L2*P[0]*P[0];
			}
		}
}

int anisotropicSmoothing(
	std::vector<float> & im
,	const ImageSize &p_imSize
){
	//declaration 
	vector<float> output,grady,gradx,imFinal;
	float myalpha = 0.001,sig=0.5,sig2=4.0;
	unsigned T = 15,rho = 4;
	int C= 1,limitXSize,limitXSizeJ;
	float stepT=10000.15;
	// %% 1 gaussian K_sigma
	float t=0.f,s=0;

	// generate mask
	vector<float> mask_sig,mask_rho;
	Generate_mask(sig,myalpha,mask_sig,limitXSize,2);
	Generate_mask(sig2,myalpha,mask_rho,limitXSizeJ,3);
	
	output.resize(im.size());

	//convolusion with mask
	Filter2d(im,output,p_imSize,mask_sig,limitXSize/2,limitXSize);
	//calcule gradient 
	
	gradientY(output,grady,p_imSize,1);
    gradientX(output,gradx,p_imSize,1);

	vector<float> Jxx,Jxy,Jyy,Jyy_n,Jxy_n,Jxx_n;
	Jxx.resize(im.size());
	Jxy.resize(im.size());
	Jyy.resize(im.size());
	Jyy_n.resize(im.size());
	Jxy_n.resize(im.size());
	Jxx_n.resize(im.size());
	for(unsigned i = 0;i<p_imSize.height;i++){
		for(unsigned j =0;j<p_imSize.width;j++){
			Jyy[i*p_imSize.width+j]=grady[i*p_imSize.width+j]*grady[i*p_imSize.width+j];
			Jxx[i*p_imSize.width+j]=gradx[i*p_imSize.width+j]*gradx[i*p_imSize.width+j];
			Jxy[i*p_imSize.width+j]=grady[i*p_imSize.width+j]*gradx[i*p_imSize.width+j];
			
			}
		}

	
	//Filter2d(Jyy,Jyy_n,p_imSize,mask_sig,limitXSize/2,limitXSize);
	//Filter2d(Jxy,Jxy_n,p_imSize,mask_sig,limitXSize/2,limitXSize);
//	Filter2d(Jxx,Jxx_n,p_imSize,mask_sig,limitXSize/2,limitXSize);



	vector<float> v2x,v2y,evec,eval,lamda1,lamda2,v1x,v1y,di,Dxx,Dyy,Dxy,pixel;
	//spectreTensor(Jxx_n,Jxy_n,Jyy_n,Dxx,Dxy,Dyy,p_imSize);


	spectreTensor2(im,Jxx,Jyy,Jxy,p_imSize);
/*
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
	for(unsigned i =0 ; i<p_imSize.height;i++){
		for(unsigned j=0;j<p_imSize.width;j++){
			A(0,0)=Jxx_n[i*p_imSize.width+j];
			A(0,1)=Jxy_n[i*p_imSize.width+j];
			A(1,0)=Jxy_n[i*p_imSize.width+j];
			A(1,1)=Jyy_n[i*p_imSize.width+j];
			//Eigen::EigenSolver<Eigen::Matrix<float, 2,2> > s(B);
			 Eigen::EigenSolver<Eigen::Matrix2d> es(A);
			 Eigen::Matrix2d D = es.pseudoEigenvalueMatrix();
			 Eigen::Matrix2d V = es.pseudoEigenvectors();
			 v2x[i*p_imSize.width+j]=V(0,1);
			 v2y[i*p_imSize.width+j]=V(1,1);
			 lamda1[i*p_imSize.width+j]=D(1,1);
			 lamda2[i*p_imSize.width+j]=D(0,0);

			 if(pow(v2x[i*p_imSize.width+j],2)+pow(v2y[i*p_imSize.width+j],2)!=0){
			 v2x[i*p_imSize.width+j]/=-sqrt(pow(v2x[i*p_imSize.width+j],2)+pow(v2y[i*p_imSize.width+j],2));
			 v2y[i*p_imSize.width+j]/=-sqrt(pow(v2x[i*p_imSize.width+j],2)+pow(v2y[i*p_imSize.width+j],2));
			 }
			 di[i*p_imSize.width+j]=lamda1[i*p_imSize.width+j]-lamda2[i*p_imSize.width+j];
			//return 0;
		}
	}


	for(unsigned i =0 ; i<p_imSize.height;i++)
		for(unsigned j=0;j<p_imSize.width;j++){
			lamda1[i*p_imSize.width+j]=myalpha +(1-myalpha)*exp(-1/(pow(di[i*p_imSize.width+j],2)));
			Dxx[i*p_imSize.width+j]=(pow(v2y[i*p_imSize.width+j],2)*lamda1[i*p_imSize.width+j])+(pow(v2x[i*p_imSize.width+j],2)*myalpha);
			Dyy[i*p_imSize.width+j]=(pow(v2x[i*p_imSize.width+j],2)*lamda1[i*p_imSize.width+j])+(pow(v2y[i*p_imSize.width+j],2)*myalpha);
			Dxy[i*p_imSize.width+j]=((v2x[i*p_imSize.width+j]*v2y[i*p_imSize.width+j])*lamda1[i*p_imSize.width+j])+(pow(v2x[i*p_imSize.width+j],2)*myalpha);
		}
*/



NonNegativityDiscretization(im,imFinal,p_imSize,Dxx,Dyy,Dxy,0.15);
	for(unsigned i=0;i<p_imSize.whc;i++)
	im[i]=imFinal[i];
	}

void initializeNlbParameters(
	nlbParams &o_paramStep1
,	nlbParams &o_paramStep2
,   const float p_sigma
,	const ImageSize &p_imSize
,	const bool p_useArea1
,   const bool p_useArea2
,   const bool p_verbose
){
	//! Standard deviation of the noise
	o_paramStep1.sigma = p_sigma;
	o_paramStep2.sigma = p_sigma;

	//! Size of patches
	if (p_imSize.nChannels == 1) {
		o_paramStep1.sizePatch = (p_sigma < 30.f ? 5 : 7);
		o_paramStep2.sizePatch = 5;
	}
	else {
		o_paramStep1.sizePatch = (p_sigma < 20.f ? 3 :
                                 (p_sigma < 50.f ? 5 : 7));
        o_paramStep2.sizePatch = (p_sigma < 50.f ? 3 :
                                 (p_sigma < 70.f ? 5 : 7));
	}

	//! Number of similar patches
	if (p_imSize.nChannels == 1) {
		o_paramStep1.nSimilarPatches =	(p_sigma < 10.f ? 35 :
										(p_sigma < 30.f ? 45 :
										(p_sigma < 80.f ? 90 : 100)));
		o_paramStep2.nSimilarPatches =	(p_sigma < 20.f ? 15 :
										(p_sigma < 40.f ? 25 :
										(p_sigma < 80.f ? 30 : 45)));
	}
	else {
		o_paramStep1.nSimilarPatches = o_paramStep1.sizePatch * o_paramStep1.sizePatch * 3;
		o_paramStep2.nSimilarPatches = o_paramStep2.sizePatch * o_paramStep2.sizePatch * 3;
	}

	//! Offset: step between two similar patches
	o_paramStep1.offSet = o_paramStep1.sizePatch / 2;
	o_paramStep2.offSet = o_paramStep2.sizePatch / 2;

	//! Use the homogeneous area detection trick
	o_paramStep1.useHomogeneousArea = p_useArea1;
	o_paramStep2.useHomogeneousArea = p_useArea2;

	//! Size of the search window around the reference patch (must be odd)
	o_paramStep1.sizeSearchWindow = o_paramStep1.nSimilarPatches / 2;
	if (o_paramStep1.sizeSearchWindow % 2 == 0) {
		o_paramStep1.sizeSearchWindow++;
	}
	o_paramStep2.sizeSearchWindow = o_paramStep2.nSimilarPatches / 2;
	if (o_paramStep2.sizeSearchWindow % 2 == 0) {
		o_paramStep2.sizeSearchWindow++;
	}

	//! Size of boundaries used during the sub division
	o_paramStep1.boundary = int(1.5f * float(o_paramStep1.sizeSearchWindow));
	o_paramStep2.boundary = int(1.5f * float(o_paramStep2.sizeSearchWindow));

	//! Parameter used to determine if an area is homogeneous
	o_paramStep1.gamma = 1.05f;
	o_paramStep2.gamma = 1.05f;

	//! Parameter used to estimate the covariance matrix
	if (p_imSize.nChannels == 1) {
		o_paramStep1.beta = (p_sigma < 15.f ? 1.1f :
                            (p_sigma < 70.f ? 1.f : 0.9f));
		o_paramStep2.beta = (p_sigma < 15.f ? 1.1f :
                            (p_sigma < 35.f ? 1.f : 0.9f));
	}
	else {
		o_paramStep1.beta = 1.f;
		o_paramStep2.beta = (p_sigma < 50.f ? 1.2f : 1.f);
	}

	//! Parameter used to determine similar patches
	o_paramStep2.tau = 16.f * o_paramStep2.sizePatch * o_paramStep2.sizePatch * p_imSize.nChannels;

	//! Print information?
	o_paramStep1.verbose = p_verbose;
	o_paramStep2.verbose = p_verbose;

	//! Is first step?
	o_paramStep1.isFirstStep = true;
	o_paramStep2.isFirstStep = false;

	//! Boost the paste trick
	o_paramStep1.doPasteBoost = true;
	o_paramStep2.doPasteBoost = true;
}

/**
 * @brief Main function to process the whole NL-Bayes algorithm.
 *
 * @param i_imNoisy: contains the noisy image;
 * @param o_imBasic: will contain the basic estimate image after the first step;
 * @param o_imFinal: will contain the final denoised image after the second step;
 * @param p_imSize: size of the image;
 * @param p_useArea1 : if true, use the homogeneous area trick for the first step;
 * @param p_useArea2 : if true, use the homogeneous area trick for the second step;
 * @param p_sigma : standard deviation of the noise;
 * @param p_verbose : if true, print some informations.
 *
 * @return EXIT_FAILURE if something wrong happens during the whole process.
 **/
int runNlBayes(
	std::vector<float> const& i_imNoisy
,   std::vector<float> &o_imBasic
,	std::vector<float> &o_imFinal
,	const ImageSize &p_imSize
,	const bool p_useArea1
,	const bool p_useArea2
,	const float p_sigma
,   const bool p_verbose
){
	//! Only 1, 3 or 4-channels images can be processed.
	const unsigned chnls = p_imSize.nChannels;
	if (! (chnls == 1 || chnls == 3 || chnls == 4)) {
		cout << "Wrong number of channels. Must be 1 or 3!!" << endl;
		return EXIT_FAILURE;
	}

	//! Number of available cores
	unsigned nbThreads = 1;
#ifdef _OPENMP
    nbThreads = omp_get_max_threads();
    if (p_verbose) {
        cout << "Open MP is used" << endl;
    }
#endif

	//! Initialization
	o_imBasic.resize(i_imNoisy.size());
	o_imFinal.resize(i_imNoisy.size());

	//! Parameters Initialization
	nlbParams paramStep1, paramStep2;
	initializeNlbParameters(paramStep1, paramStep2, p_sigma, p_imSize, p_useArea1, p_useArea2,
                         p_verbose);

	//! Step 1
	if (paramStep1.verbose) {
		cout << "1st Step...";
	}

	//! RGB to YUV
	vector<float> imNoisy = i_imNoisy;
	transformColorSpace(imNoisy, p_imSize, true);
	//! Divide the noisy image into sub-images in order to easier parallelize the process
	const unsigned nbParts = 2 * nbThreads;
	vector<vector<float> > imNoisySub(nbParts), imBasicSub(nbParts), imFinalSub(nbParts);
	ImageSize imSizeSub;
	if (subDivide(imNoisy, imNoisySub, p_imSize, imSizeSub, paramStep1.boundary, nbParts)
		!= EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

	//! Process all sub-images
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, nbParts/nbThreads) \
            shared(imNoisySub, imBasicSub, imFinalSub, imSizeSub) \
		firstprivate (paramStep1)
#endif
	for (int n = 0; n < (int) nbParts; n++) {
	    processNlBayes(imNoisySub[n], imBasicSub[n], imFinalSub[n], imSizeSub, paramStep1);
	}

	//! Get the basic estimate
	if (subBuild(o_imBasic, imBasicSub, p_imSize, imSizeSub, paramStep1.boundary)
		!= EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

	//! YUV to RGB
	transformColorSpace(o_imBasic, p_imSize, false);

	if (paramStep1.verbose) {
		cout << "done." << endl;
	}

	//! 2nd Step
	if (paramStep2.verbose) {
		cout << "2nd Step...";
	}

	//! Divide the noisy and basic images into sub-images in order to easier parallelize the process
	if (subDivide(i_imNoisy, imNoisySub, p_imSize, imSizeSub, paramStep2.boundary, nbParts)
        != EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}
	if (subDivide(o_imBasic, imBasicSub, p_imSize, imSizeSub, paramStep2.boundary, nbParts)
		!= EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

	//! Process all sub-images
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, nbParts/nbThreads) \
            shared(imNoisySub, imBasicSub, imFinalSub) \
		firstprivate (paramStep2)
#endif
	for (int n = 0; n < (int) nbParts; n++) {
		processNlBayes(imNoisySub[n], imBasicSub[n], imFinalSub[n], imSizeSub, paramStep2);
	}

	//! Get the final result
	if (subBuild(o_imFinal, imFinalSub, p_imSize, imSizeSub, paramStep2.boundary)
		!= EXIT_SUCCESS) {
		return EXIT_FAILURE;
	}

	if (paramStep2.verbose) {
		cout << "done." << endl << endl;
	}

	return EXIT_SUCCESS;
}

/**
 * @brief Generic step of the NL-Bayes denoising (could be the first or the second).
 *
 * @param i_imNoisy: contains the noisy image;
 * @param io_imBasic: will contain the denoised image after the first step (basic estimation);
 * @param o_imFinal: will contain the denoised image after the second step;
 * @param p_imSize: size of i_imNoisy;
 * @param p_params: see nlbParams.
 *
 * @return none.
 **/
void processNlBayes(
	std::vector<float> const& i_imNoisy
,	std::vector<float> &io_imBasic
,	std::vector<float> &o_imFinal
,	const ImageSize &p_imSize
,	nlbParams &p_params
){
	//! Parameters initialization
	const unsigned sW		= p_params.sizeSearchWindow;
	const unsigned sP		= p_params.sizePatch;
	const unsigned sP2		= sP * sP;
	const unsigned sPC		= sP2 * p_imSize.nChannels;
	const unsigned nSP		= p_params.nSimilarPatches;
	unsigned nInverseFailed	= 0;
	const float threshold	= p_params.sigma * p_params.sigma * p_params.gamma *
                                (p_params.isFirstStep ? p_imSize.nChannels : 1.f);

	//! Allocate Sizes
	if (p_params.isFirstStep) {
		io_imBasic.resize(p_imSize.whc);
	}
	o_imFinal.resize(p_imSize.whc);

	//! Used matrices during Bayes' estimate
	vector<vector<float> > group3d(p_imSize.nChannels, vector<float> (nSP * sP2));
	vector<float> group3dNoisy(sW * sW * sPC), group3dBasic(sW * sW * sPC);
	vector<unsigned> index(p_params.isFirstStep ? nSP : sW * sW);
	matParams mat;
	mat.group3dTranspose.resize(p_params.isFirstStep ? nSP * sP2 : sW * sW * sPC);
	mat.tmpMat          .resize(p_params.isFirstStep ? sP2 * sP2 : sPC * sPC);
	mat.baricenter      .resize(p_params.isFirstStep ? sP2 : sPC);
	mat.covMat          .resize(p_params.isFirstStep ? sP2 * sP2 : sPC * sPC);
	mat.covMatTmp       .resize(p_params.isFirstStep ? sP2 * sP2 : sPC * sPC);

	//! ponderation: weight sum per pixel
	vector<float> weight(i_imNoisy.size(), 0.f);

	//! Mask: non-already processed patches
	vector<bool> mask(p_imSize.wh, false);

	//! Only pixels of the center of the image must be processed (not the boundaries)
	for (unsigned i = sW; i < p_imSize.height - sW; i++) {
		for (unsigned j = sW; j < p_imSize.width - sW; j++) {
			mask[i * p_imSize.width + j] = true;
		}
	}

	for (unsigned ij = 0; ij < p_imSize.wh; ij += p_params.offSet) {
		//! Only non-seen patches are processed
		if (mask[ij]) {
			//! Search for similar patches around the reference one
			unsigned nSimP = p_params.nSimilarPatches;
			if (p_params.isFirstStep) {
				estimateSimilarPatchesStep1(i_imNoisy, group3d, index, ij, p_imSize, p_params);
			}
			else {
				nSimP = estimateSimilarPatchesStep2(i_imNoisy, io_imBasic, group3dNoisy,
					group3dBasic, index, ij, p_imSize, p_params);
			}

			//! Initialization
			bool doBayesEstimate = true;

			//! If we use the homogeneous area trick
			if (p_params.useHomogeneousArea) {
				if (p_params.isFirstStep) {
					doBayesEstimate = !computeHomogeneousAreaStep1(group3d, sP, nSP,
						threshold, p_imSize);
				}
				else {
					doBayesEstimate = !computeHomogeneousAreaStep2(group3dNoisy, group3dBasic,
						sP, nSimP, threshold, p_imSize);
				}
			}

			//! Else, use Bayes' estimate
			if (doBayesEstimate) {
				if (p_params.isFirstStep) {
					computeBayesEstimateStep1(group3d, mat, nInverseFailed, p_params);
				}
				else {
					computeBayesEstimateStep2(group3dNoisy, group3dBasic, mat, nInverseFailed,
						p_imSize, p_params, nSimP);
				}
			}

			//! Aggregation
			if (p_params.isFirstStep) {
				computeAggregationStep1(io_imBasic, weight, mask, group3d, index, p_imSize,
					p_params);
			}
			else {
				computeAggregationStep2(o_imFinal, weight, mask, group3dBasic, index, p_imSize,
					p_params, nSimP);
			}
		}
	}

	//! Weighted aggregation
	computeWeightedAggregation(i_imNoisy, io_imBasic, o_imFinal, weight, p_params, p_imSize);

	if (nInverseFailed > 0 && p_params.verbose) {
		cout << "nInverseFailed = " << nInverseFailed << endl;
	}
}

/**
 * @brief Estimate the best similar patches to a reference one.
 *
 * @param i_im: contains the noisy image on which distances are processed;
 * @param o_group3d: will contain values of similar patches;
 * @param o_index: will contain index of similar patches;
 * @param p_ij: index of the reference patch;
 * @param p_imSize: size of the image;
 * @param p_params: see processStep1 for more explanation.
 *
 * @return none.
 **/
void estimateSimilarPatchesStep1(
	std::vector<float> const& i_im
,	std::vector<std::vector<float> > &o_group3d
,	std::vector<unsigned> &o_index
,	const unsigned p_ij
,	const ImageSize &p_imSize
,	const nlbParams &p_params
){
	//! Initialization
	const unsigned sW		= p_params.sizeSearchWindow;
	const unsigned sP		= p_params.sizePatch;
	const unsigned width	= p_imSize.width;
	const unsigned chnls	= p_imSize.nChannels;
	const unsigned wh		= width * p_imSize.height;
	const unsigned ind		= p_ij - (sW - 1) * (width + 1) / 2;
	const unsigned nSimP	= p_params.nSimilarPatches;
	vector<pair<float, unsigned> > distance(sW * sW);

	//! Compute distance between patches
	for (unsigned i = 0; i < sW; i++) {
		for (unsigned j = 0; j < sW; j++) {
			const unsigned k = i * width + j + ind;
			float diff = 0.f;
			float soma = 0.f;
			float somb = 0.f;

			for (unsigned p = 0; p < sP; p++) {
				for (unsigned q = 0; q < sP; q++) {
					//diff += tmpValue * tmpValue;
								float somc = 0.f;
								float somd = 0.f;
						for (unsigned p1 = 0; p1 < sP; p1++) {
							for (unsigned q1 = 0; q1 < sP; q1++) {
									somc += i_im[p_ij + p1 * width + q1];
									somd += i_im[k + p1 * width + q1];
						}
					}
					somc = (1/(sP*sP))*somc;
					somd = (1/(sP*sP))*somd;
					const float tmpValue = (i_im[p_ij + p * width + q]-somc) * (i_im[k + p * width + q]-somd);

					diff += tmpValue;
					//	soma += pow(i_im[p_ij + p * width + q] ,2);
					//	somb += pow(i_im[k + p * width + q],2);
						soma += pow((i_im[p_ij + p * width + q]-somc)  ,2);
						somb += pow((i_im[k + p * width + q]-somd),2);
				}
			}
			diff = diff / (sqrt(soma*somb));
			//! Save all distances
			distance[i * sW + j] = make_pair(diff, k);
		}
	}

	//! Keep only the N2 best similar patches
	partial_sort(distance.begin(), distance.begin() + nSimP, distance.end(), comparaisonFirst);

	//! Register position of patches
	for (unsigned n = 0; n < nSimP; n++) {
		o_index[n] = distance[n].second;
	}

	//! Register similar patches into the 3D group
	for (unsigned c = 0; c < chnls; c++) {
		for (unsigned p = 0, k = 0; p < sP; p++) {
			for (unsigned q = 0; q < sP; q++) {
				for (unsigned n = 0; n < nSimP; n++, k++) {
					o_group3d[c][k] = i_im[o_index[n] + p * width + q + c * wh];
				}
			}
		}
	}
}

/**
 * @brief Keep from all near patches the similar ones to the reference patch for the second step.
 *
 * @param i_imNoisy: contains the original noisy image;
 * @param i_imBasic: contains the basic estimation;
 * @param o_group3dNoisy: will contain similar patches for all channels of i_imNoisy;
 * @param o_group3dBasic: will contain similar patches for all channels of i_imBasic;
 * @param o_index: will contain index of similar patches;
 * @param p_ij: index of the reference patch;
 * @param p_imSize: size of images;
 * @param p_params: see processStep2 for more explanations.
 *
 * @return number of similar patches kept.
 **/
unsigned estimateSimilarPatchesStep2(
	std::vector<float> const& i_imNoisy
,	std::vector<float> const& i_imBasic
,	std::vector<float> &o_group3dNoisy
,	std::vector<float> &o_group3dBasic
,	std::vector<unsigned> &o_index
,	const unsigned p_ij
,	const ImageSize &p_imSize
,	const nlbParams &p_params
){
	//! Initialization
	const unsigned width	= p_imSize.width;
	const unsigned chnls	= p_imSize.nChannels;
	const unsigned wh		= width * p_imSize.height;
	const unsigned sP		= p_params.sizePatch;
	const unsigned sW		= p_params.sizeSearchWindow;
	const unsigned ind		= p_ij - (sW - 1) * (width + 1) / 2;
	vector<pair<float, unsigned> > distance(sW * sW);

	//! Compute distance between patches
	for (unsigned i = 0; i < sW; i++) {
		for (unsigned j = 0; j < sW; j++) {
			const unsigned k = i * width + j + ind;
			float diff = 0.0f;

			for (unsigned c = 0; c < chnls; c++) {
				const unsigned dc = c * wh;
				for (unsigned p = 0; p < sP; p++) {
					for (unsigned q = 0; q < sP; q++) {
						const float tmpValue = i_imBasic[dc + p_ij + p * width + q]
											- i_imBasic[dc + k + p * width + q];
						diff += tmpValue * tmpValue;
					}
				}
			}

			//! Save all distances
			distance[i * sW + j] = make_pair(diff, k);
		}
	}

	//! Keep only the nSimilarPatches best similar patches
	partial_sort(distance.begin(), distance.begin() + p_params.nSimilarPatches, distance.end(),
		comparaisonFirst);

	//! Save index of similar patches
	const float threshold = (p_params.tau > distance[p_params.nSimilarPatches - 1].first ?
							p_params.tau : distance[p_params.nSimilarPatches - 1].first);
	unsigned nSimP = 0;

	//! Register position of similar patches
	for (unsigned n = 0; n < distance.size(); n++) {
		if (distance[n].first < threshold) {
			o_index[nSimP++] = distance[n].second;
		}
	}

	//! Save similar patches into 3D groups
	for (unsigned c = 0, k = 0; c < chnls; c++) {
		for (unsigned p = 0; p < sP; p++) {
			for (unsigned q = 0; q < sP; q++) {
				for (unsigned n = 0; n < nSimP; n++, k++) {
					o_group3dNoisy[k] = i_imNoisy[c * wh + o_index[n] + p * width + q];
					o_group3dBasic[k] = i_imBasic[c * wh + o_index[n] + p * width + q];
				}
			}
		}
	}

	return nSimP;
}

/**
 * @brief Detect if we are in an homogeneous area. In this case, compute the mean.
 *
 * @param io_group3d: contains for each channels values of similar patches. If an homogeneous area
 *			is detected, will contain the average of all pixels in similar patches;
 * @param p_sP2: size of each patch (sP x sP);
 * @param p_nSimP: number of similar patches;
 * @param p_threshold: threshold below which an area is declared homogeneous;
 * @param p_doLinearRegression: if true, apply a linear regression to average value of pixels;
 * @param p_imSize: size of the image.
 *
 * @return 1 if an homogeneous area is detected, 0 otherwise.
 **/
int computeHomogeneousAreaStep1(
	std::vector<std::vector<float> > &io_group3d
,	const unsigned p_sP
,	const unsigned p_nSimP
,	const float p_threshold
,	const ImageSize &p_imSize
){
	//! Initialization
	const unsigned N = p_sP * p_sP * p_nSimP;

	//! Compute the standard deviation of the set of patches
	float stdDev = 0.f;
	for (unsigned c = 0; c < p_imSize.nChannels; c++) {
		stdDev += computeStdDeviation(io_group3d[c], p_sP * p_sP, p_nSimP, 1);
	}

	//! If we are in an homogeneous area
	if (stdDev < p_threshold) {
		for (unsigned c = 0; c < p_imSize.nChannels; c++) {
            float mean = 0.f;

            for (unsigned k = 0; k < N; k++) {
                mean += io_group3d[c][k];
            }

            mean /= (float) N;

            for (unsigned k = 0; k < N; k++) {
                io_group3d[c][k] = mean;
            }
        }
		return 1;
	}
	else {
		return 0;
	}
}

/**
 * @brief Detect if we are in an homogeneous area. In this case, compute the mean.
 *
 * @param io_group3dNoisy: contains values of similar patches for the noisy image;
 * @param io_group3dBasic: contains values of similar patches for the basic image. If an homogeneous
 *		area is detected, will contain the average of all pixels in similar patches;
 * @param p_sP2: size of each patch (sP x sP);
 * @param p_nSimP: number of similar patches;
 * @param p_threshold: threshold below which an area is declared homogeneous;
 * @param p_imSize: size of the image.
 *
 * @return 1 if an homogeneous area is detected, 0 otherwise.
 **/
int computeHomogeneousAreaStep2(
	std::vector<float> const& i_group3dNoisy
,	std::vector<float> &io_group3dBasic
,	const unsigned p_sP
,	const unsigned p_nSimP
,	const float p_threshold
,	const ImageSize &p_imSize
){
	//! Parameters
	const unsigned sP2	= p_sP * p_sP;
	const unsigned sPC = sP2 * p_imSize.nChannels;

	//! Compute the standard deviation of the set of patches
	const float stdDev = computeStdDeviation(i_group3dNoisy, sP2, p_nSimP, p_imSize.nChannels);

	//! If we are in an homogeneous area
	if (stdDev < p_threshold) {
		for (unsigned c = 0; c < p_imSize.nChannels; c++) {
            float mean = 0.f;

            for (unsigned n = 0; n < p_nSimP; n++) {
                for (unsigned k = 0; k < sP2; k++) {
                    mean += io_group3dBasic[n * sPC + c * sP2 + k];
                }
            }

            mean /= float(sP2 * p_nSimP);

            for (unsigned n = 0; n < p_nSimP; n++) {
                for (unsigned k = 0; k < sP2; k++) {
                    io_group3dBasic[n * sPC + c * sP2 + k] = mean;
                }
            }
		}
		return 1;
	}
	else {
		return 0;
	}
}

/**
 * @brief Compute the Bayes estimation.
 *
 * @param io_group3d: contains all similar patches. Will contain estimates for all similar patches;
 * @param i_mat: contains :
 *		- group3dTranspose: allocated memory. Used to contain the transpose of io_group3dNoisy;
 *		- baricenter: allocated memory. Used to contain the baricenter of io_group3dBasic;
 *		- covMat: allocated memory. Used to contain the covariance matrix of the 3D group;
 *		- covMatTmp: allocated memory. Used to process the Bayes estimate;
 *		- tmpMat: allocated memory. Used to process the Bayes estimate;
 * @param io_nInverseFailed: update the number of failed matrix inversion;
 * @param p_params: see processStep1 for more explanation.
 *
 * @return none.
 **/
 void computeBayesEstimateStep1(
	std::vector<std::vector<float> > &io_group3d
,	matParams &i_mat
,	unsigned &io_nInverseFailed
,	nlbParams &p_params
){
	//! Parameters
	const unsigned chnls = io_group3d.size();
	const unsigned nSimP = p_params.nSimilarPatches;
	const unsigned sP2   = p_params.sizePatch * p_params.sizePatch;
	const float valDiag  = p_params.beta * p_params.sigma * p_params.sigma;

	//! Bayes estimate
	for (unsigned c = 0; c < chnls; c++) {

	    //! Center data around the baricenter
		centerData(io_group3d[c], i_mat.baricenter, nSimP, sP2);

		//! Compute the covariance matrix of the set of similar patches
		covarianceMatrix(io_group3d[c], i_mat.covMat, nSimP, sP2);

		//! Bayes' Filtering
		if (inverseMatrix(i_mat.covMat, sP2) == EXIT_SUCCESS) {
            productMatrix(i_mat.group3dTranspose, i_mat.covMat, io_group3d[c], sP2, sP2, nSimP);
            for (unsigned k = 0; k < sP2 * nSimP; k++) {
                io_group3d[c][k] -= valDiag * i_mat.group3dTranspose[k];
            }
		}
		else {
			io_nInverseFailed++;
		}

		//! Add baricenter
		for (unsigned j = 0, k = 0; j < sP2; j++) {
			for (unsigned i = 0; i < nSimP; i++, k++) {
			    io_group3d[c][k] += i_mat.baricenter[j];
			}
		}
	}
}

/**
 * @brief Compute the Bayes estimation.
 *
 * @param i_group3dNoisy: contains all similar patches in the noisy image;
 * @param io_group3dBasic: contains all similar patches in the basic image. Will contain estimates
 *			for all similar patches;
 * @param i_mat: contains :
 *		- group3dTranspose: allocated memory. Used to contain the transpose of io_group3dNoisy;
 *		- baricenter: allocated memory. Used to contain the baricenter of io_group3dBasic;
 *		- covMat: allocated memory. Used to contain the covariance matrix of the 3D group;
 *		- covMatTmp: allocated memory. Used to process the Bayes estimate;
 *		- tmpMat: allocated memory. Used to process the Bayes estimate;
 * @param io_nInverseFailed: update the number of failed matrix inversion;
 * @param p_imSize: size of the image;
 * @param p_params: see processStep2 for more explanations;
 * @param p_nSimP: number of similar patches.
 *
 * @return none.
 **/
void computeBayesEstimateStep2(
	std::vector<float> &i_group3dNoisy
,	std::vector<float> &io_group3dBasic
,	matParams &i_mat
,	unsigned &io_nInverseFailed
,	const ImageSize &p_imSize
,	nlbParams p_params
,	const unsigned p_nSimP
){
	//! Parameters initialization
	const float diagVal = p_params.beta * p_params.sigma * p_params.sigma;
	const unsigned sPC  = p_params.sizePatch * p_params.sizePatch * p_imSize.nChannels;

	//! Center 3D groups around their baricenter
	centerData(io_group3dBasic, i_mat.baricenter, p_nSimP, sPC);
	centerData(i_group3dNoisy, i_mat.baricenter, p_nSimP, sPC);

	//! Compute the covariance matrix of the set of similar patches
	covarianceMatrix(io_group3dBasic, i_mat.covMat, p_nSimP, sPC);

	//! Bayes' Filtering
    for (unsigned k = 0; k < sPC; k++) {
        i_mat.covMat[k * sPC + k] += diagVal;
    }

	//! Compute the estimate
	if (inverseMatrix(i_mat.covMat, sPC) == EXIT_SUCCESS) {
        productMatrix(io_group3dBasic, i_mat.covMat, i_group3dNoisy, sPC, sPC, p_nSimP);
        for (unsigned k = 0; k < sPC * p_nSimP; k++) {
            io_group3dBasic[k] = i_group3dNoisy[k] - diagVal * io_group3dBasic[k];
        }
	}
	else {
		io_nInverseFailed++;
	}

	//! Add baricenter
	for (unsigned j = 0, k = 0; j < sPC; j++) {
		for (unsigned i = 0; i < p_nSimP; i++, k++) {
			io_group3dBasic[k] += i_mat.baricenter[j];
		}
	}
}

/**
 * @brief Aggregate estimates of all similar patches contained in the 3D group.
 *
 * @param io_im: update the image with estimate values;
 * @param io_weight: update corresponding weight, used later in the weighted aggregation;
 * @param io_mask: update values of mask: set to true the index of an used patch;
 * @param i_group3d: contains estimated values of all similar patches in the 3D group;
 * @param i_index: contains index of all similar patches contained in i_group3d;
 * @param p_imSize: size of io_im;
 * @param p_params: see processStep1 for more explanation.
 *
 * @return none.
 **/
void computeAggregationStep1(
	std::vector<float> &io_im
,	std::vector<float> &io_weight
,	std::vector<bool> &io_mask
,	std::vector<std::vector<float> > const& i_group3d
,	std::vector<unsigned> const& i_index
,	const ImageSize &p_imSize
,	const nlbParams &p_params
){
	//! Parameters initializations
	const unsigned chnls	= p_imSize.nChannels;
	const unsigned width	= p_imSize.width;
	const unsigned height	= p_imSize.height;
	const unsigned sP		= p_params.sizePatch;
	const unsigned nSimP	= p_params.nSimilarPatches;

	//! Aggregate estimates
	for (unsigned n = 0; n < nSimP; n++) {
		const unsigned ind = i_index[n];
		for (unsigned c = 0; c < chnls; c++) {
			const unsigned ij = ind + c * width * height;
			for (unsigned p = 0; p < sP; p++) {
				for (unsigned q = 0; q < sP; q++) {
					io_im[ij + p * width + q] += i_group3d[c][(p * sP + q) * nSimP + n];
					io_weight[ij + p * width + q]++;
				}
			}
		}

		//! Use Paste Trick
		io_mask[ind] = false;

		if (p_params.doPasteBoost) {
			io_mask[ind - width ] = false;
			io_mask[ind + width ] = false;
			io_mask[ind - 1		] = false;
			io_mask[ind + 1		] = false;
		}
	}
}

/**
 * @brief Aggregate estimates of all similar patches contained in the 3D group.
 *
 * @param io_im: update the image with estimate values;
 * @param io_weight: update corresponding weight, used later in the weighted aggregation;
 * @param io_mask: update values of mask: set to true the index of an used patch;
 * @param i_group3d: contains estimated values of all similar patches in the 3D group;
 * @param i_index: contains index of all similar patches contained in i_group3d;
 * @param p_imSize: size of io_im;
 * @param p_params: see processStep2 for more explanation;
 * @param p_nSimP: number of similar patches.
 *
 * @return none.
 **/
void computeAggregationStep2(
	std::vector<float> &io_im
,	std::vector<float> &io_weight
,	std::vector<bool> &io_mask
,	std::vector<float> const& i_group3d
,	std::vector<unsigned> const& i_index
,	const ImageSize &p_imSize
,	const nlbParams &p_params
,	const unsigned p_nSimP
){
	//! Parameters initializations
	const unsigned chnls	= p_imSize.nChannels;
	const unsigned width	= p_imSize.width;
	const unsigned wh		= width * p_imSize.height;
	const unsigned sP		= p_params.sizePatch;

	//! Aggregate estimates
	for (unsigned n = 0; n < p_nSimP; n++) {
		const unsigned ind = i_index[n];
		for (unsigned c = 0, k = 0; c < chnls; c++) {
			const unsigned ij = ind + c * wh;
			for (unsigned p = 0; p < sP; p++) {
				for (unsigned q = 0; q < sP; q++, k++) {
					io_im[ij + p * width + q] += i_group3d[k * p_nSimP + n];
					io_weight[ij + p * width + q]++;
				}
			}
		}

		//! Apply Paste Trick
		io_mask[ind] = false;

		if (p_params.doPasteBoost) {
			io_mask[ind - width ] = false;
			io_mask[ind + width ] = false;
			io_mask[ind - 1     ] = false;
			io_mask[ind + 1     ] = false;
		}
	}
}

/**
 * @brief Compute the final weighted aggregation.
 *
 * i_imReference: image of reference, when the weight if null;
 * io_imResult: will contain the final image;
 * i_weight: associated weight for each estimate of pixels.
 *
 * @return : none.
 **/
void computeWeightedAggregation(
	std::vector<float> const& i_imNoisy
,	std::vector<float> &io_imBasic
,	std::vector<float> &io_imFinal
,	std::vector<float> const& i_weight
,	const nlbParams &p_params
,	const ImageSize &p_imSize
){
	for (unsigned c = 0, k = 0; c < p_imSize.nChannels; c++) {

		for (unsigned ij = 0; ij < p_imSize.wh; ij++, k++) {

			//! To avoid weighting problem (particularly near boundaries of the image)
			if (i_weight[k] > 0.f) {
				if (p_params.isFirstStep) {
					io_imBasic[k] /= i_weight[k];
				}
				else {
					io_imFinal[k] /= i_weight[k];
				}
			}
			else {
				if (p_params.isFirstStep) {
					io_imBasic[k] = i_imNoisy[k];
				}
				else {
					io_imFinal[k] = io_imBasic[k];
				}
			}
		}
	}
}
