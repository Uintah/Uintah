//////////////////////////////////////////////////////////////////////
// Image.cpp - Load and save images of many file formats.
//
// by David K. McAllister, 1997-1998.

#include <Malloc/Allocator.h>

#include <Rendering/ZTex/Image.h>

#include <Rendering/ZTex/Assert.h>
#include <Rendering/ZTex/Utils.h>

#include <stdio.h>

#include <fstream>
using namespace std;

#include <stdlib.h>

namespace SemotusVisum {
namespace Rendering {

// Convert the number of image channels in this image.
void Image::SetChan(const int ch)
{
	if(ch < 1 || ch > 4) return;

	unsigned char *P2 = doSetChan(ch, Pix);

	delete [] Pix;
	Pix = P2;

	chan = ch;
	size = wid * hgt;
	dsize = wid * hgt * chan;
}

// Rescales the image to the given size.
void Image::Resize(const int w, const int h)
{
  wid = w;
  hgt = h;
  size = wid * hgt;
  dsize = wid * hgt * chan;
}

// Return a new pixel buffer based on P, but with ch channels.
unsigned char *Image::doSetChan(const int ch, unsigned char *P)
{
	int i, i2;

	int dsize1 = wid * hgt * ch;

	// Change the image's parameters.
	unsigned char *P2 = scinew unsigned char[dsize1];
	
	if(ch == chan)
	{
		memcpy(P2, P, dsize1);
		return P2;
	}

	// Change the number of channels.
	switch(chan)
	{
	case 1:
		{
			switch(ch)
			{
			case 2:
				for(i=i2=0; i<dsize; i++)
				{
					P2[i2++] = P[i];
					P2[i2++] = 255;
				}
				break;
			case 3:
				for(i=i2=0; i<dsize; i++)
				{
					P2[i2++] = P[i];
					P2[i2++] = P[i];
					P2[i2++] = P[i];
				}
				break;
			case 4:
				for(i=i2=0; i<dsize; i++)
				{
					P2[i2++] = P[i];
					P2[i2++] = P[i];
					P2[i2++] = P[i];
					P2[i2++] = 255;
				}
				break;
			}
		}
		break;
	case 2:
		{
			switch(ch)
			{
			case 1:
				for(i=i2=0; i<dsize; i+=2)
				{
					P2[i2++] = P[i];
				}
				break;
			case 3:
				for(i=i2=0; i<dsize; i+=2)
				{
					P2[i2++] = P[i];
					P2[i2++] = P[i];
					P2[i2++] = P[i];
				}
				break;
			case 4:
				for(i=i2=0; i<dsize; i++)
				{
					P2[i2++] = P[i];
					P2[i2++] = P[i];
					P2[i2++] = P[i];
					P2[i2++] = P[++i];
				}
				break;
			}
		}
		break;
	case 3:
		{
			switch(ch)
			{
			case 1:
				for(i=i2=0; i<dsize; )
				{
					P2[i2++] = (unsigned char)((77 * int(P[i++]) + 150 * int(P[i++]) +
						29 * int(P[i++])) >> 8);
				}
				break;
			case 2:
				for(i=i2=0; i<dsize; )
				{
					P2[i2++] = (unsigned char)((77 * int(P[i++]) + 150 * int(P[i++]) +
						29 * int(P[i++])) >> 8);
					P2[i2++] = 255;
				}
				break;
			case 4:
				for(i=i2=0; i<dsize; )
				{
					P2[i2++] = P[i++];
					P2[i2++] = P[i++];
					P2[i2++] = P[i++];
					P2[i2++] = 255;
				}
				break;
			}
		}
		break;
	case 4:
		{
			switch(ch)
			{
			case 1:
				for(i=i2=0; i<dsize; i++)
				{
					P2[i2++] = (unsigned char)((77 * int(P[i++]) + 150 * int(P[i++]) +
						29 * int(P[i++])) >> 8);
				}
				break;
			case 2:
				for(i=i2=0; i<dsize; )
				{
					P2[i2++] = (unsigned char)((77 * int(P[i++]) + 150 * int(P[i++]) +
						29 * int(P[i++])) >> 8);
					P2[i2++] = P[i++];
				}
				break;
			case 3:
				for(i=i2=0; i<dsize; i++)
				{
					P2[i2++] = P[i++];
					P2[i2++] = P[i++];
					P2[i2++] = P[i++];
				}
				break;
			}
		}
		break;
	}

	return P2;
}

// Convert this image to the given parameters.
void Image::Set(const int _w, const int _h, const int _ch,
				const bool init)
{
	int w = _w, h = _h, ch = _ch;

	if(w < 0) w = wid;
	if(h < 0) h = hgt;
	if(ch < 0) ch = chan;
	
	int dsize1 = w * h * ch;

	if(Pix)
	{
		// Deal with existing picture.
		if(dsize1 <= 0)
		{
			delete [] Pix;
			Pix = NULL;
			wid = hgt = chan = size = dsize = 0;
			return;
		}
		
		// Copies the pixels to a new array of the right num. color channels.
		unsigned char *P2 = doSetChan(ch, Pix);
		delete [] Pix;
		Pix = P2;
		chan = ch;
		dsize = wid * hgt * chan;
		size = wid * hgt;

		// Rescale to the new width and height.
		Resize(w, h);
	}
	else
	{
		if(dsize1 > 0)
		{
			wid = w;
			hgt = h;
			chan = ch;
			dsize = dsize1;
			size = wid * hgt;
			Pix = scinew unsigned char[dsize];
			if(init)
				fill(0);
		}
		else
		{
			Pix = NULL;
			wid = hgt = chan = size = dsize = 0;
		}
	}
}

// Convert this image to the given parameters.
void Image::Set(const Image &Img, const int _w, const int _h,
				const int _ch, const bool init)
{
	int w = _w, h = _h, ch = _ch;

	if(w < 0) w = Img.wid;
	if(h < 0) h = Img.hgt;
	if(ch < 0) ch = Img.chan;
	
	int dsize1 = w * h * ch;
	
	if(Pix)
		delete [] Pix;
	
	// Deal with existing picture.
	if(dsize1 <= 0)
	{
		wid = hgt = chan = size = dsize = 0;
		Pix = NULL;
		return;
	}
	
	if(Img.Pix)
	{		
		// Copies the pixels to a new array of the right num. color channels.
		wid = Img.wid;
		hgt = Img.hgt;
		chan = Img.chan;
		size = wid * hgt;
		dsize = wid * hgt * chan;

		Pix = doSetChan(ch, Img.Pix);
		chan = ch;
		dsize = wid * hgt * chan;

		// Rescale to the new width and height.
		Resize(w, h);
	}
	else
	{
		if(dsize1 > 0)
		{
			wid = w;
			hgt = h;
			chan = ch;
			dsize = dsize1;
			size = wid * hgt;
			Pix = scinew unsigned char[dsize];
			if(init)
				fill(0);
		}
	}
}

Image Image::operator+(const Image &Im) const
{
	ASSERT(chan == Im.chan && wid == Im.wid && hgt == Im.hgt);
	
	Image Out(wid, hgt, chan);
	
	for(int i=0; i<dsize; i++)
	{
		unsigned short sum = Pix[i] + Im.Pix[i];
		Out.Pix[i] = (sum <= 255) ? sum : 255;
	}
	
	return Out;
}

Image &Image::operator+=(const Image &Im)
{
	ASSERT(chan == Im.chan && wid == Im.wid && hgt == Im.hgt);
	
	for(int i=0; i<dsize; i++)
	{
		unsigned short sum = Pix[i] + Im.Pix[i];
		Pix[i] = (sum <= ((unsigned char)255)) ? sum : ((unsigned char)255);
	}
	
	return *this;
}

// Copy channel number src of image img to channel dest of this image.
// Things work fine if img is *this.
void Image::SpliceChan(const Image &Im, const int src, const int dest)
{
	ASSERT(wid == Im.wid && hgt == Im.hgt);
	ASSERT(src < Im.chan && dest < chan);

	int tind = dest;
	int sind = src;
	for(; tind < dsize ; )
		Pix[tind += chan] = Im.Pix[sind += Im.chan];
}

void Image::VFlip()
{
  int lsize = wid * chan;
  unsigned char *tbuf = scinew unsigned char[lsize];

  for(int y=0; y<hgt/2; y++)
	{
	  memcpy(tbuf, &Pix[y*lsize], lsize);
	  memcpy(&Pix[y*lsize], &Pix[(hgt-y-1)*lsize], lsize);
	  memcpy(&Pix[(hgt-y-1)*lsize], tbuf, lsize);
	}
  delete [] tbuf;
}

} // namespace Tools
} // namespace Remote

