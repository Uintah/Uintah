#ifndef _range_image_h
#define _range_image_h

#include <Packages/Remote/Tools/Image/Image.h>
#include <Packages/Remote/Tools/Util/Utils.h>
#include <Packages/Remote/Tools/Math/MiscMath.h>

#include <iostream>
#include <fstream>

namespace Remote {
using namespace std;
class RangeImage : public Image
{
public:
	inline float *FPix() const
	{
		return (float *)Pix;
	}
	
	inline float &operator()(int x, int y)
	{
		float *F = FPix();
		return F[y*wid+x];
	}
	
	inline float &operator[](int x)
	{
		float *F = FPix();
		return F[x];
	}

	inline ~RangeImage() {}

	inline RangeImage() : Image() {}
  
	inline RangeImage(const int w, const int h, const bool fill = false) : Image(w, h, 4, fill) {}
	
	// Read a PFM file.
	inline RangeImage(const char *FName)
	{
	  ifstream InFile(FName, ios::in | ios::binary);
	  if(!InFile.is_open())
		{
			cerr << "Failed to open PFM file `" << FName << "'.\n";
		}
		
		char Magic1, Magic2;
		InFile >> Magic1 >> Magic2;
		
		if(Magic1 != 'P' || Magic2 != '7')
		{
			cerr << FName << " is not a known PFM file.\n";
			InFile.close();
		}
		
		InFile.get();
		char c = InFile.peek();
		while(c == '#')
		{
			char line[999];
			InFile.getline(line, 1000);
			cerr << line << endl;
			c = InFile.peek();
		}
		
		int dyn_range;
		InFile >> wid >> hgt >> dyn_range;
		InFile.get();
		
		if(dyn_range != 255)
		{
			cerr << "Must be 255. Was " << dyn_range << endl;
		}
		
		chan = 4;
		size = wid * hgt;
		dsize = size * chan;
		float *F = new float[size];
		Pix = (unsigned char *)F;
		
		InFile.read((char *)Pix, dsize);
		
#ifdef SCI_LITTLE_ENDIAN
		// Intel is little-endian.
		// Always assume they are stored as big-endian.
		ConvertLong((unsigned int *)Pix, size);
#endif
		
		InFile.close();
	}
		
	// Hooks the given image into this image object.
	virtual inline void SetImage(float *p, const int w, const int h, const int ch = 4)
	{
		Image::SetImage((unsigned char *)p, w, h, ch);
	}

	// Save the float image as a PFM file.
	bool SavePFM(const char *FName) const
	{
		if(Pix == NULL || chan < 1 || wid < 1 || hgt < 1)
		{
			cerr << "Image is not defined. Not saving.\n";
			return true;
		}
		
		if(chan != 4)
		{
			cerr << "Can't save a " << chan << " channel image as a PFM.\n";
			return true;
		}
		
#if defined ( SCI_MACHINE_sgi ) || defined ( SCI_MACHINE_hp )
		ofstream OF(FName);
		if(!OF.rdbuf()->is_open())
#else
			ofstream OF(FName, ios::out | ios::binary);
		if(!OF.is_open())
#endif
		{
			cerr << "Failed to open file `" << FName << "'.\n";
			return true;
		}
		
		OF << "P7\n" << wid << " " << hgt << endl << 255 << endl;
		
#ifdef SCI_LITTLE_ENDIAN
		// Need to convert floats to big-endian before saving.
		char *Tmp = new char[dsize];
		memcpy(Tmp, Pix, dsize);
		ConvertLong((unsigned int *)Tmp, size);
		OF.write((char *)Tmp, dsize);
		delete [] Tmp;
#else
		OF.write((char *)Pix, dsize);
#endif 
		OF.close();
		
		cerr << "Wrote PFM file " << FName << endl;
		
		return false;
	}
	
	// Uses floats
	// Bilinearly sample the exact spot.
	// Returns flag = true if any sample is bogus.
	inline float bilerp1f(const float x, const float y, bool &flag, float VThresh = 1e33) const
	{
		float *F = FPix();
		int x0 = int(x); float xt = x - float(x0);
		int y0 = int(y); float yt = y - float(y0);
		
		int ind = y0 * wid + x0;
		float b00 = F[ind];
		float b01 = F[ind+1];
		float b10 = F[ind+wid];
		float b11 = F[ind+wid+1];
		
		float d0 = (b01 - b00);
		float d1 = (b11 - b10);
		float b0 = xt * d0 + b00;
		float b1 = xt * d1 + b10;
		float d = (b1 - b0);
		float b = yt * d + b0;
		
		// Tell if any of the points are too far away.
		flag = (b00 >= VThresh) || (b01 >= VThresh) ||
			(b10 >= VThresh) || (b11 >= VThresh);
		
		return b;
	}
	
	// Uses floats
	// Bilinearly sample the exact spot.
	// Returns flag = true if any sample is bogus.
	inline float bilerp(const float x, const float y) const
	{
		float *F = FPix();
		int x0 = int(x); float xt = x - float(x0);
		int y0 = int(y); float yt = y - float(y0);
		
		int ind = y0 * wid + x0;
		float b00 = F[ind];
		float b01 = F[ind+1];
		float b10 = F[ind+wid];
		float b11 = F[ind+wid+1];
		
		float d0 = (b01 - b00);
		float d1 = (b11 - b10);
		float b0 = xt * d0 + b00;
		float b1 = xt * d1 + b10;
		float d = (b1 - b0);
		float b = yt * d + b0;
		
		return b;
	}
	
	// Uses doubles
	// Bilinearly sample the exact spot.
	// Returns flag = true if any sample is bogus.
	inline double bilerpd(const double x, const double y) const
	{
		float *F = FPix();
		int x0 = int(x); double xt = x - double(x0);
		int y0 = int(y); double yt = y - double(y0);
		
		int ind = y0 * wid + x0;
		float b00 = F[ind];
		float b01 = F[ind+1];
		float b10 = F[ind+wid];
		float b11 = F[ind+wid+1];
		
		double d0 = (b01 - b00);
		double d1 = (b11 - b10);
		double b0 = xt * d0 + b00;
		double b1 = xt * d1 + b10;
		double d = (b1 - b0);
		double b = yt * d + b0;
		
		return b;
	}
	
	// Map the range image to colors and save it as a GIF.
	inline void SaveConv(char *fname, float ScaleFac = 1.0f)
	{
		Image Bob(wid, hgt, 1);
		unsigned char *P = Bob.Pix;
		float *F = FPix();
		
		for(int i=0; i<size; i++)
		{
			P[i] = (unsigned char)Clamp(0.0, (F[i] * ScaleFac + 0.5f), 255.0);
		}
		
		Bob.SaveGIF(fname);
	}

	// Push peaks to be flatter.
	void DeSpeckle(float Thresh = 1.0f);
	
	// Blur the float image with a gaussian kernel stored in a 5x5.
	void Blur(double stdev = 2.5);
};

} // End namespace Remote


#endif
