//////////////////////////////////////////////////////////////////////
// Filter.cpp - Stuff to filter images.
// Copyright 1998 by David K. McAllister
//////////////////////////////////////////////////////////////////////

#include <Packages/Remote/Tools/Util/Assert.h>
#include <Packages/Remote/Tools/Image/Image.h>
#include <Packages/Remote/Tools/Image/RangeImage.h>
#include <Packages/Remote/Tools/Math/MiscMath.h>

#include <iostream>
using namespace std;

namespace Remote {
// Shamelessly copied from Aly Ray Smith's Principles of Image Compositing
#define INT_MULT(a, b, t) ((t) = (a) * (b) + 0x80, ((((t)>>8) + (t))>>8))

inline int int_mult(unsigned char a, unsigned char b)
{
	int t = a * b + 0x80;
	return ((t>>8) + t);
}

// 3x3 blur with special kernel. Assumes single channel image.
void Image::FastBlur1()
{
	ASSERT(chan == 1);
	// cerr << "FastBlur1\n";
	int y;
	
	// Allocates space for image.
	unsigned char *P2 = new unsigned char[dsize];
	
	// Do corners.
	{
		unsigned short C = Pix[0] << 2;
		C += Pix[1] << 2;
		C += Pix[wid] << 2;
		C += Pix[wid+1] << 1;
		C += Pix[wid+1];
		P2[0] = (unsigned char)((C + 16) / 15);

		C = Pix[wid-1] << 2;
		C += Pix[wid-2] << 2;
		C += Pix[wid+wid-1] << 2;
		C += Pix[wid+wid-2] << 1;
		C += Pix[wid+wid-2];
		P2[wid-1] = (unsigned char)((C + 16) / 15);
		
		int ib=(hgt-1)*wid;
		C = Pix[ib] << 2;
		C += Pix[ib+1] << 2;
		C += Pix[ib-wid] << 2;
		C += Pix[ib-wid+1] << 1;
		C += Pix[ib-wid+1];
		P2[ib] = (unsigned char)((C + 16) / 15);

		C = Pix[ib+wid-1] << 2;
		C += Pix[ib+wid-2] << 2;
		C += Pix[ib-1] << 2;
		C += Pix[ib-2] << 1;
		C += Pix[ib-2];
		P2[ib+wid-1] = (unsigned char)((C + 16) / 15);
	}

	// Do top and bottom edges.
	int it=1, ib=(hgt-1)*wid+1;
	for(; it<wid-1; ib++, it++)
	{
		// Top
		unsigned short C = Pix[it] << 2;
		C += Pix[it+1] << 2;
		C += Pix[it-1] << 2;
		C += Pix[it+wid] << 2;
		C += Pix[it+wid+1];
		C += Pix[it+wid+1] << 1;
		C += Pix[it+wid-1];
		C += Pix[it+wid-1] << 1;
		P2[it] = (unsigned char)((C + 16) / 22);
		
		// Bottom
		C = Pix[ib] << 2;
		C += Pix[ib+1] << 2;
		C += Pix[ib-1] << 2;
		C += Pix[ib-wid] << 2;
		C += Pix[ib-wid+1];
		C += Pix[ib-wid+1] << 1;
		C += Pix[ib-wid-1];
		C += Pix[ib-wid-1] << 1;
		P2[ib] = (unsigned char)((C + 16) / 22);
		//P2[ib] = 255;
	}
	
	for(y=1; y<hgt-1; y++)
	{
		int il = y*wid, ir = y*wid+wid-1;
		
		// Left side
		unsigned short C = Pix[il] << 2;
		C += Pix[il+1] << 2;
		C += Pix[il+wid] << 2;
		C += Pix[il-wid] << 2;
		C += Pix[il+wid+1];
		C += Pix[il+wid+1] << 1;
		C += Pix[il-wid+1];
		C += Pix[il-wid+1] << 1;
		P2[il] = (unsigned char)((C + 16) / 22);
		
		// Right side
		C = Pix[ir] << 2;
		C += Pix[ir-1] << 2;
		C += Pix[ir+wid] << 2;
		C += Pix[ir-wid] << 2;
		C += Pix[ir+wid-1];
		C += Pix[ir+wid-1] << 1;
		C += Pix[ir-wid-1];
		C += Pix[ir-wid-1] << 1;
		P2[ir] = (unsigned char)((C + 16) / 22);
		//P2[ir] = 255;
	}
	
#ifdef SCI_USE_MP
#pragma parallel 
#pragma pfor schedtype(gss) local(y)
#endif
	for(y=1; y<hgt-1; y++)
	{
		int ind = y*wid+1;
		for(int x=1; x<wid-1; x++, ind++)
		{
			// Sum of weights: 343 444 343 = 32
			unsigned short C = Pix[ind] << 2;
			C += Pix[ind+1] << 2;
			C += Pix[ind-1] << 2;
			C += Pix[ind+wid] << 2;
			C += Pix[ind-wid] << 2;
			C += Pix[ind+wid+1];
			C += Pix[ind+wid+1] << 1;
			C += Pix[ind+wid-1];
			C += Pix[ind+wid-1] << 1;
			C += Pix[ind-wid+1];
			C += Pix[ind-wid+1] << 1;
			C += Pix[ind-wid-1];
			C += Pix[ind-wid-1] << 1;
			P2[ind] = (unsigned char)((C + 16) >> 5);
		}
	}

	// Hook the new image into this Image.
	delete[] Pix;
	Pix = P2;
}

// N is the size of ONE DIMENSION of the kernel.
// Assumes an odd kernel size. Assumes single channel image.
void Image::Filter1(const int N, const KERTYPE *kernel)
{
	ASSERT(chan == 1);

	int N2 = N/2, y, x;
	
	// Allocates space for image.
	unsigned char *P2 = new unsigned char[dsize];

	// Do top and bottom edges.
	{
	for(int x=N2; x<wid-N2; x++)
	{
		for(y=0; y<N2; y++)
			P2[y*wid+x] = SampleSlow1(x, y, N, kernel);
		for(y=hgt-N2; y<hgt; y++)
			P2[y*wid+x] = SampleSlow1(x, y, N, kernel);
	}
	}

	// Do left and right edges.
	for(y=0; y<hgt; y++)
	{
		for(x=0; x<N2; x++)
			P2[y*wid+x] = SampleSlow1(x, y, N, kernel);
		for(x=wid-N2; x<wid; x++)
			P2[y*wid+x] = SampleSlow1(x, y, N, kernel);
	}

#ifdef SCI_USE_MP
#pragma parallel 
#pragma pfor schedtype(gss) local(y)
#endif
	for(y=N2; y<hgt-N2; y++)
	{
		int y0 = y-N2;
		int y1 = y+N2;
		for(int x=N2; x<wid-N2; x++)
		{
			// Add the pixels that contribute to this one.
			int x0 = x-N2;
			int x1 = x+N2;
			
			unsigned int C = 0;
			int ker = 0;
			for(int yy=y0; yy <= y1; yy++)
			{
				for(int xx=x0; xx <= x1; xx++, ker++)
				{
					C += Pix[yy*wid+xx] * kernel[ker];
				}
			}
			P2[y*wid+x] = (unsigned char)((C + 0x8000) >> 16);
		}
	}

	// Hook the new image into this Image.
	delete[] Pix;
	Pix = P2;
}

void Image::Filter3(const int N, const KERTYPE *kernel)
{
	ASSERT(chan == 3);

	int N2 = N/2, x, y;
	
	// Allocates space for color image.
	unsigned char *P2 = new unsigned char[dsize];

	Pixel *Pp2 = (Pixel *)P2;
	// Do top and bottom edges.
	for(x=N2; x<wid-N2; x++)
	{
		for(y=0; y<N2; y++)
			Pp2[y*wid+x] = SampleSlow3(x, y, N, kernel);
		for(y=hgt-N2; y<hgt; y++)
			Pp2[y*wid+x] = SampleSlow3(x, y, N, kernel);
	}

	// Do left and right edges.
	for(y=0; y<hgt; y++)
	{
		for(x=0; x<N2; x++)
			Pp2[y*wid+x] = SampleSlow3(x, y, N, kernel);
		for(x=wid-N2; x<wid; x++)
			Pp2[y*wid+x] = SampleSlow3(x, y, N, kernel);
	}

	for(y=N2; y<hgt-N2; y++)
	{
		int y0 = y-N2;
		int y1 = y+N2;
		for(x=N2; x<wid-N2; x++)
		{
			// Add the pixels that contribute to this one.
			int x0 = x-N2;
			int x1 = x+N2;
			
			unsigned int Cr = 0, Cg = 0, Cb = 0;
			int ker = 0;
			//KERTYPE SK = 0;
			for(int yy=y0; yy <= y1; yy++)
			{
				for(int xx=x0; xx <= x1; xx++, ker++)
				{
					Cr += Pix[(yy*wid+xx)*3] * kernel[ker];
					Cg += Pix[(yy*wid+xx)*3+1] * kernel[ker];
					Cb += Pix[(yy*wid+xx)*3+2] * kernel[ker];
				}
			}
			P2[(y*wid+x)*3] = (unsigned char)((Cr + 0x8000) >> 16);
			P2[(y*wid+x)*3+1] = (unsigned char)((Cg + 0x8000) >> 16);
			P2[(y*wid+x)*3+2] = (unsigned char)((Cb + 0x8000) >> 16);
		}
	}

	// Hook the new image into this Image.
	delete[] Pix;
	Pix = P2;
}

// N is the size of ONE DIMENSION of the kernel.
// Assumes an odd kernel size.
void Image::Filter(const int N, const KERTYPE *kernel)
{
	if(chan == 1)
		Filter1(N, kernel);
	else if(chan == 3)
		Filter3(N, kernel);
	else
		cerr << "Filtering not supported on " << chan << " channel images.\n";
}

double *MakeBlurKernel(const int N, const double sigma)
{
	double *kernel = new double[N*N];
	
	int N2 = N/2, x, y;
	
	double S = 0;
	for(y= -N2; y<=N2; y++)
	{
		for(x= -N2; x<=N2; x++)
		{
			double G = Gaussian2(x, y, sigma);
			kernel[(y + N2)*N + (x+N2)] = G;
			
			S += G;
		}
	}
	
	// normalize the kernel.
	for(y = 0; y<N; y++)
		for(x = 0; x<N; x++)
			kernel[y*N + x] /= S;

	return kernel;
}

// Make a fixed point kernel from a double kernel.
// Does NOT delete the old kernel.
KERTYPE *DoubleKernelToFixed(const int N, double *dkernel)
{
	KERTYPE *ckernel = new KERTYPE[N*N];

	// I should multiply by 256 and clamp to 255 but I won't.
	double SD = 0;
	double SC = 0;
	for(int i=0; i<N*N; i++)
	{
		ckernel[i] = (KERTYPE)(65535.0 * dkernel[i] + 0.5);
		SD += dkernel[i];
		SC += ckernel[i];
	}

	//cerr << "Double kernel weight = " << SD << endl;
	//cerr << "Byte kernel weight = " << int(SC) << endl;

	return ckernel;
}

// Blur this image with a kernel of size N and st. dev. of sigma.
// sigma = 1 seems to work well for different N.
// Makes a gaussian of st. dev. sigma, samples it at the places on
// the kernel (1 pixel is 1 unit), and then normalizes the kernel
// to have unit mass.
void Image::Blur(const int N, const double sig)
{
	if(chan == 1 && sig == 0)
	{
		FastBlur1();
		return;
	}

	double sigma = sig;
	if(sig < 0)
		sigma = double(N) / 3.0;
	
	double *dkernel = MakeBlurKernel(N, sigma);
	KERTYPE *ckernel = DoubleKernelToFixed(N, dkernel);
	delete [] dkernel;

	Filter(N, ckernel);

	delete [] ckernel;
}

// Upsample linearly.
void Image::HorizFiltLinear(unsigned char *Ld, int wd,
							unsigned char *Ls, int ws)
{
	int xs = 0, xsw = 0, xdw = 0;
	for(int xd=0; xd<wd; xd++, xdw += wd)
	{
		// Sample from xs and xs+1 into xd.
		for(int c=0; c<chan; c++)
		{
			Ld[xd] = (Ls[xs] * (xdw-xsw)) / ws +
				(Ls[xs+1] * (xsw+ws-xdw)) / ws;
		}
		
		// Maybe step xs.
		if(xsw+ws < xdw)
		{
			xs++;
			xsw += ws;
		}
	}
}

// Rescales the image to the given size.
void Image::Resize(const int w, const int h)
{
#if 0
  if(w < 1 || h < 1 || chan < 1)
    {
      wid = hgt = chan = size = dsize = 0;
      if(Pix)
	delete [] Pix;
      return;
    }

  if(size < 1 || Pix == NULL)
    return;

  unsigned char *P;
  int x, y;

				// Scale the width first.
  int dsize1 = w * hgt * chan;

  if(w == wid)
    P = Pix;
  else
    {
      P = new unsigned char[dsize1];
      if(w > wid)
	{
				// Upsample using cubic filter.
	  for(y=0; y<hgt; y++)
	    HorizFiltLinear(&P[y*w*chan], w, &Pix[y*wid*chan], wid);
	}
      else
	{
				// Downsample using gaussian.
	  for(y=0; y<hgt; y++)
	    HorizFiltLinear(&P[y*w*chan], w, &Pix[y*wid*chan], wid);
	  //HorizFiltGaussian(&P[y*w*chan], w, &Pix[y*wid*chan], wid);
	}
    }

  if(P != Pix)
    delete [] Pix;
  Pix = P;
  wid = w;

				// Scale the height.
  dsize1 = wid * h * chan;

  if(h == hgt)
    P = Pix;
  else
    {
      P = new unsigned char[dsize1];
      if(h > hgt)
	{
				// Upsample using cubic filter.
	  for(x=0; x<wid; x++)
	    VertFiltLinear(&P[x*chan], h, &Pix[x*chan], hgt, wid*chan);
	  //VertFiltCubic(&P[x*chan], h, &Pix[x*chan], hgt, wid*chan);
	}
      else
	{
				// Downsample using gaussian.
	  for(x=0; x<wid; x++)
	    VertFiltLinear(&P[x*chan], h, &Pix[x*chan], hgt, wid*chan);
	  //VertFiltGaussian(&P[x*chan], h, &Pix[x*chan], hgt, wid*chan);
	}
    }

  if(P != Pix)
    delete [] Pix;
  Pix = P;
#endif
  wid = w;
  hgt = h;
  size = wid * hgt;
  dsize = wid * hgt * chan;
}

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
} // End namespace Remote


