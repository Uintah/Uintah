//////////////////////////////////////////////////////////////////////
// Image.h - Generic image class suitable for graphics and stuff.
//
// Copyright 1997 by David K. McAllister
//
//////////////////////////////////////////////////////////////////////

#ifndef _image_h
#define _image_h

#include <string.h>

// A 1 channel image is luminance (grayscale).
// A 2 channel image is luminance and alpha.
// A 3 channel image is RGB.
// A 4 channel image is RGBA.

namespace SemotusVisum {
namespace Rendering {

//----------------------------------------------------------------------
				// This is a not very useful struct
				// for RGB pixels.
struct Pixel {
  unsigned char r, g, b;
  
  inline Pixel(const unsigned char _r, const unsigned char _g,
    const unsigned char _b) {
    r = _r; g = _g; b = _b;
  }
  
  inline Pixel() {}
};

//----------------------------------------------------------------------
class Image
{
  unsigned char *doSetChan(const int ch, unsigned char *P);
  void HorizFiltLinear(unsigned char *Ld, int wd,
    unsigned char *Ls, int ws);
  
				// I recommend unsigned int for the
				// kernel, storing a fixed point real
				// number in 16 bits. This is because
				// we sum the k*p into an int, so we
				// need to worry about overflow.
#define KERTYPE unsigned int
  
  void FastBlur1();
  
				// Sample the given pixel in the
				// image, using kernel of size N as an
				// aperture. Slow because it takes
				// edges into account.
  inline unsigned char SampleSlow1(const int x, const int y, const int N,
    const KERTYPE *kernel) const;
  
				// Sample the given pixel in the
				// image, using kernel of size N as an
				// aperture. Slow because it takes
				// edges into account.
  inline Pixel SampleSlow3(const int x, const int y, const int N,
    const KERTYPE *kernel) const;
  
  void Filter1(const int N, const KERTYPE *kernel);
  void Filter3(const int N, const KERTYPE *kernel);
  
public:
  unsigned char *Pix;
  int wid, hgt, chan, size, dsize;
  
				// Constructors
  inline Image();
				// Create an image of the specified
				// kind.
  inline Image(const int w, const int h,
    const int ch = 3, const bool init = false);
  
				// Copy constructor.
  inline Image(const Image &Img, const int w = -1, const int h = -1,
    const int ch = -1, const bool init = false);
  
				// Load an image from a file. Specify
				// the number of channels desired.
  Image(const char *FName, const int ch = -1);
  
				// Destroy an image.
  inline ~Image();
  
				// Access functions
  
  inline Pixel & operator() (const int x, const int y) const;
  
				// Returns this pixel as a Pixel.
  inline Pixel & operator[] (const int i) const;
  
				// Returns this pixel as a byte (can
				// be used as lvalue.)
  inline unsigned char &ch(const int x, const int y) const;
  
				// Returns a pointer to the start of
				// this pixel.
  inline unsigned char *chp(const int x, const int y) const;
  
				// Returns a pointer to the start of
				// this pixel.
  inline unsigned char *chp(const int i) const;
  
				// Uses unsigned chars Bilinearly
				// sample the exact spot.
  inline unsigned char bilerp1uc(const float x, const float y) const;
  
				// Uses floats Bilinearly sample the
				// exact spot.
  inline float bilerp1(const float x, const float y) const;
  
				// Uses floats. Returns a color pixel.
				// Bilinearly sample the exact spot.
  inline Pixel bilerp(const float x, const float y) const;
	
				// Functions for returning the pixel
				// pointer in different formats.
  inline unsigned char *CPix() const ;
  inline Pixel *PPix() const;

				// Set this image to be empty.
  inline void Wipe();

				// Change the parameters of this
				// image.
  void Set(const int w, const int h, const int ch = 3,
    const bool init = false);

				// Copy the given image to this one,
				// then change this one's parameters.
  void Set(const Image &Img, const int w = -1, const int h = -1,
    const int ch = -1, const bool init = false);
	
				// Hooks the given image into this
				// image object.
  inline void SetImage(unsigned char *p, const int w, const int h, const int ch = 3);

  inline Image & operator= (const Image &Img);

  Image operator+(const Image &Im) const;
  Image &operator+=(const Image &Im);

				// Input Output Functions

  bool LoadGIF(const char *fname);
  bool LoadPPM(const char *fname);
  bool LoadRas(const char *fname);
  bool LoadRGB(const char *fname);
#ifdef SCI_USE_TIFF
  bool LoadTIFF(const char *fname);
#endif
#ifdef SCI_USE_JPEG
  bool LoadJPEG(const char *fname);
#endif

				// Detects type from filename. Won't
				// modify number of channels.  Return
				// false on success.
  bool Save(const char *FName) const;

  bool SaveGIF(const char *FName) const;
  bool SavePPM(const char *FName) const;
#ifdef SCI_USE_TIFF
  bool SaveTIFF(const char *FName) const;
#endif
#ifdef SCI_USE_JPEG
  bool SaveJPEG(const char *FName) const;
#endif
	
				// Utility functions

  void Resize(const int w, const int h);
  void SetChan(const int ch);

				// Copy channel number src of image
				// img to channel dest of this image.
				// Things work fine if img is *this.
				// Channel numbers start with 0.
  void SpliceChan(const Image &img, const int src, const int dest);

				// Flip this image vertically.
  void VFlip();

				// The size of the filter kernel and
				// the st. dev. of the gaussian.  -1
				// means to use sigma = N/3.
  void Blur(const int N, const double sigma = -1);

  void Filter(const int N, const KERTYPE *kernel);
  friend double *MakeBlurKernel(const int N, const double sigma);
  friend KERTYPE *DoubleKernelToFixed(const int N, double *kernel);

  inline void fill(const Pixel p);
  inline void fill(const unsigned char p);
  
};

//----------------------------------------------------------------------
				// Sample the given pixel in the
				// image, using kernel of size N as an
				// aperture. Slow because it takes
				// edges into account.
unsigned char Image::SampleSlow1(const int x, const int y,
  const int N, const KERTYPE *kernel) const
{
  int N2 = N/2;
  KERTYPE KS = 0, C = 0;
  
  for(int ky=0; ky<N; ky++) {
    int yp = y + ky - N2;
    
    if(yp >= 0 && yp < hgt) {
      for(int kx=0; kx<N; kx++) {
	int xp = x + kx - N2;
	if(xp >= 0 && xp < wid) {
	  KS += kernel[ky*N+kx];
	  C += Pix[yp*wid+xp] * kernel[ky*N+kx];
	}
      }
    }
  }
  
  return (unsigned char)(C / KS);
}

//----------------------------------------------------------------------
				// Sample the given pixel in the
				// image, using kernel of size N as an
				// aperture. Slow because it takes
// edges into account.
Pixel Image::SampleSlow3(const int x, const int y, const int N,
  const KERTYPE *kernel) const {
  Pixel *P = PPix();
  int N2 = N/2;
  unsigned int Cr = 0, Cg = 0, Cb = 0;
  KERTYPE KS = 0;
    
  for(int ky=0; ky<N; ky++) {
    int yp = y + ky - N2;
	
    if(yp >= 0 && yp < hgt) {
      for(int kx=0; kx<N; kx++) {
	int xp = x + kx - N2;
	if(xp >= 0 && xp < wid) {
	  KS += kernel[ky*N+kx];
	  Cr += P[yp*wid+xp].r * kernel[ky*N+kx];
	  Cg += P[yp*wid+xp].g * kernel[ky*N+kx];
	  Cb += P[yp*wid+xp].b * kernel[ky*N+kx];
	}
      }
    }
  }
    
  return Pixel((unsigned char)(Cr / KS),
    (unsigned char)(Cg / KS), (unsigned char)(Cb / KS));
}
  

// Constructors
Image::Image() {
  Pix = NULL;
  wid = hgt = size = dsize = chan = 0;
}
  
// Create an image of the specified
// kind.
Image::Image(const int w, const int h,
  const int ch, const bool init) {
  Pix = NULL;
  Set(w, h, ch, init);
}
  
// Copy constructor.
Image::Image(const Image &Img, const int w, const int h,
  const int ch, const bool init)	{
  Pix = NULL;
  Set(Img, w, h, ch, init);
}
  
				// Destroy an image.
Image::~Image() {
  if(Pix)
    delete [] Pix;
}
  
// Access functions
  
Pixel & Image::operator() (const int x, const int y) const {
  return ((Pixel *)Pix)[y*wid+x];
}
  
// Returns this pixel as a Pixel.
Pixel & Image::operator[] (const int i) const {
  return ((Pixel *)Pix)[i];
}
  
// Returns this pixel as a byte (can
// be used as lvalue.)
unsigned char &Image::ch(const int x, const int y) const {
  return Pix[(y*wid+x)*chan];
}
  
// Returns a pointer to the start of
// this pixel.
unsigned char *Image::chp(const int x, const int y) const {
  return &Pix[(y*wid+x)*chan];
}
  
// Returns a pointer to the start of
// this pixel.
unsigned char *Image::chp(const int i) const {
  return &Pix[i*chan];
}
  
// Uses unsigned chars Bilinearly
// sample the exact spot.
unsigned char Image::bilerp1uc(const float x, const float y) const {
  int x0 = int(x); float xt = x - float(x0);
  int y0 = int(y); float yt = y - float(y0);
  int xs = int(xt * 65536.0);
  int ys = int(yt * 65536.0);
    
  int ind = y0 * wid + x0;
  unsigned char b00 = Pix[ind];
  unsigned char b01 = Pix[ind+1];
  unsigned char b10 = Pix[ind+wid];
  unsigned char b11 = Pix[ind+wid+1];
    
  int b0 = xs * (b01 - b00) + (b00 << 16);
  int b1 = xs * (b11 - b10) + (b10 << 16);
    
  return (ys * ((b1 - b0) >> 16) + b0) >> 16;
}
  
// Uses floats Bilinearly sample the
// exact spot.
float Image::bilerp1(const float x, const float y) const {
  int x0 = int(x); float xt = x - float(x0);
  int y0 = int(y); float yt = y - float(y0);
    
  int ind = y0 * wid + x0;
  float b00 = Pix[ind];
  float b01 = Pix[ind+1];
  float b10 = Pix[ind+wid];
  float b11 = Pix[ind+wid+1];
    
  float b0 = xt * (b01 - b00) + b00;
  float b1 = xt * (b11 - b10) + b10;
  float b = yt * (b1 - b0) + b0;
  return b;
}
  
// Uses floats. Returns a color pixel.
// Bilinearly sample the exact spot.
Pixel Image::bilerp(const float x, const float y) const {
  int x0 = int(x); float xt = x - float(x0);
  int y0 = int(y); float yt = y - float(y0);
    
  int ind = 3 * (y0 * wid + x0);
  int indw3 = ind + 3 * wid;
    
  Pixel P;
    
  {
    float b00 = Pix[ind];
    float b01 = Pix[ind+3];
    float b10 = Pix[indw3];
    float b11 = Pix[indw3+3];
      
    float b0 = xt * (b01 - b00) + b00;
    float b1 = xt * (b11 - b10) + b10;
    P.r = (unsigned char)(yt * (b1 - b0) + b0);
  }
		
  {
    float b00 = Pix[ind+1];
    float b01 = Pix[ind+4];
    float b10 = Pix[indw3+1];
    float b11 = Pix[indw3+4];
			
    float b0 = xt * (b01 - b00) + b00;
    float b1 = xt * (b11 - b10) + b10;
    P.g = (unsigned char)(yt * (b1 - b0) + b0);
  }
		
  {
    float b00 = Pix[ind+2];
    float b01 = Pix[ind+5];
    float b10 = Pix[indw3+2];
    float b11 = Pix[indw3+5];
			
    float b0 = xt * (b01 - b00) + b00;
    float b1 = xt * (b11 - b10) + b10;
    P.b = (unsigned char)(yt * (b1 - b0) + b0);
  }

  return P;
}
	
// Functions for returning the pixel
// pointer in different formats.
unsigned char *Image::CPix() const {
  return Pix;
}

inline Pixel *Image::PPix() const {
  return (Pixel *)Pix;
}

// Set this image to be empty.
void Image::Wipe() {
  Set(0, 0);
}
	
// Hooks the given image into this
// image object.
void Image::SetImage(unsigned char *p, const int w, const int h, const int ch) {
  Pix = p;
  wid = w;
  hgt = h;
  chan = ch;
  size = wid * hgt;
  dsize = size * ch;
}

Image & Image::operator= (const Image &Img) {
  Set(Img);
  return (*this);
}

void Image::fill(const Pixel p)
{
  for(int y = 0; y < size; y++)
    ((Pixel *)Pix)[y] = p;
}
	
void Image::fill(const unsigned char p)
{
  memset(Pix, p, dsize);
}


} // namespace Tools
} // namespace Remote

#endif
