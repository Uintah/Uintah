//////////////////////////////////////////////////////////////////////
// Filter.cpp - Stuff to filter images.
//
// Copyright 1998 by David K. McAllister
//
//////////////////////////////////////////////////////////////////////

/*
#include <Util/Assert.h>
#include <Image/Image.h>
#include <Image/RangeImage.h>
#include <Math/MiscMath.h>
*/

#include <iostream>
using namespace std;

// Shamelessly copied from Aly Ray Smith's Principles of Image Compositing
#define INT_MULT(a, b, t) ((t) = (a) * (b) + 0x80, ((((t)>>8) + (t))>>8))

inline int int_mult(unsigned char a, unsigned char b)
{
	int t = a * b + 0x80;
	return ((t>>8) + t);
}

namespace Remote {
namespace Modules {


void RangeImage::Blur(double stdev)
	{
		double *Kernel = MakeBlurKernel(5, stdev);
		float *F = FPix();
		
		float *N = new float[size];
		
		// Just don't blur the edges.
		memcpy(N, F, wid * sizeof(float));
		memcpy(&N[wid], &F[wid], wid * sizeof(float));
		memcpy(&N[(hgt-2)*wid], &F[(hgt-2)*wid], wid * sizeof(float));
		memcpy(&N[(hgt-1)*wid], &F[(hgt-1)*wid], wid * sizeof(float));
		
		for(int y=2; y<hgt-2; y++)
		{
			N[y*wid] = F[y*wid];
			N[y*wid+1] = F[y*wid+1];
			N[(y+1)*wid-1] = F[(y+1)*wid-1];
			N[(y+1)*wid-2] = F[(y+1)*wid-2];
			
			for(int x=2; x<wid-2; x++)
			{
				double G = 0;
				
				G += Kernel[0] * F[(y-2)*wid+x-2];
				G += Kernel[1] * F[(y-2)*wid+x-1];
				G += Kernel[2] * F[(y-2)*wid+x];
				G += Kernel[3] * F[(y-2)*wid+x+1];
				G += Kernel[4] * F[(y-2)*wid+x+2];
				
				G += Kernel[5] * F[(y-1)*wid+x-2];
				G += Kernel[6] * F[(y-1)*wid+x-1];
				G += Kernel[7] * F[(y-1)*wid+x];
				G += Kernel[8] * F[(y-1)*wid+x+1];
				G += Kernel[9] * F[(y-1)*wid+x+2];
				
				G += Kernel[10] * F[(y)*wid+x-2];
				G += Kernel[11] * F[(y)*wid+x-1];
				G += Kernel[12] * F[(y)*wid+x];
				G += Kernel[13] * F[(y)*wid+x+1];
				G += Kernel[14] * F[(y)*wid+x+2];
				
				G += Kernel[15] * F[(y+1)*wid+x-2];
				G += Kernel[16] * F[(y+1)*wid+x-1];
				G += Kernel[17] * F[(y+1)*wid+x];
				G += Kernel[18] * F[(y+1)*wid+x+1];
				G += Kernel[19] * F[(y+1)*wid+x+2];
				
				G += Kernel[20] * F[(y+2)*wid+x-2];
				G += Kernel[21] * F[(y+2)*wid+x-1];
				G += Kernel[22] * F[(y+2)*wid+x];
				G += Kernel[23] * F[(y+2)*wid+x+1];
				G += Kernel[24] * F[(y+2)*wid+x+2];
				
				N[y*wid+x] = G;
			}
		}
		
		delete [] Pix;
		Pix = (unsigned char *) N;
		
		delete [] Kernel;
	}

// If all my neighbors are > thresh above me,
// or all are < thresh below me, set me to
// their average.
void RangeImage::DeSpeckle(float Thresh)
{
  float *F = FPix();

  int ct = 0;

  for(int y=1; y<hgt-1; y++)
    {
      for(int x=1; x<wid-1; x++)
	{
	  bool loop;
	  do
	    {
	      loop = false;
	      int i = y*wid+x;
	      float &me = F[i];
	      
	      int below = 0, above = 0;
	      
	      float D;
	      D = F[i-wid-1] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i-wid] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i-wid+1] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i-1] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i+1] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i+wid-1] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i+wid] - me; below += D < -Thresh; above += D > Thresh;
	      D = F[i+wid+1] - me; below += D < -Thresh; above += D > Thresh;
	      
	      if(above == 8) {me += Thresh; loop = true;}
	      if(below == 8) {me -= Thresh; loop = true; ct++;}
	    }
	  while(loop);
	}
    }
  
  cerr << "Modified " << ct << " pixels.\n";
}

}
}

