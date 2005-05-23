/*
  PNGImage.cc

  Code to read png images was derived from libpng.  Licence for the
  code used as a base can be found at the end of the file.

*/

#include <Packages/rtrt/Core/PNGImage.h>
#include <Core/Exceptions/InternalError.h>
//#include <Packages/rtrt/Core/cexcept.h>
#include <Core/Persistent/PersistentSTL.h>

// This is for HAVE_PNG
#include <sci_defs/image_defs.h>

#ifdef HAVE_PNG
#include <Core/Thread/Mutex.h>
#include <png.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace std;

PNGImage::PNGImage(const string& s, bool flip) 
  : flipped_(flip) 
{ 
  valid_ = read_image(s.c_str()); //read also the alpha mask
}

PNGImage::PNGImage(int nu, int nv, bool flip) 
  : width_(nu), height_(nv), valid_(false), flipped_(flip) 
{
  image_.resize(width_*height_);
  alpha_.resize(width_*height_);
}

void PNGImage::get_dimensions_and_data(Array2<rtrt::Color> &c,
				       Array2<float> &d, int &nu, int &nv) {
  if (valid_) {
    c.resize(width_+2,height_+2);  // image size + slop for interpolation
    d.resize(width_+2, height_+2);
    nu=width_;
    nv=height_;
    for (unsigned v=0; v<height_; ++v)
      for (unsigned u=0; u<width_; ++u)
	{
	  c(u,v)=image_[v*width_+u];
	  d(u,v)=alpha_[v*width_+u];
	}
    
  } else {
    c.resize(0,0);
    d.resize(0,0);
    nu=0;
    nv=0;
  }
}

#ifdef HAVE_PNG

// Because of the use of all these static variables we need to make
// sure that we don't try to load in more than one image at a time.
static SCIRun::Mutex read_image_lock("PNGImage::read_image lock");

//static png_const_charp msg;

//static png_structp png_ptr = NULL;
//static png_infop info_ptr = NULL;



static void
png_cexcept_error(png_structp png_ptr, png_const_charp msg)
{
  if(png_ptr)
    ;
#ifndef PNG_NO_CONSOLE_IO
  fprintf(stderr, "libpng error: %s\n", msg);
#endif
  {
    throw SCIRun::InternalError(msg);
  }
}



bool
PNGImage::read_image(const char* filename)
{
  read_image_lock.lock();
  FILE *pfFile = NULL;

  png_byte pbSig[8];
  int iBitDepth;
  int iColorType;
 
  png_uint_32 piWidth, piHeight;
  png_color *pBkgColor = NULL;
  
  png_color_16 *pBackground = NULL;
  png_uint_32 ulChannels;
  png_uint_32 ulRowBytes;
  //image data
  png_byte** ppbImageData = NULL;
  png_byte*  pbImageData = *ppbImageData;
  static png_byte **ppbRowPointers = NULL;

  int i;

  png_structp png_ptr;
  png_infop info_ptr;

  if((pfFile = fopen(filename, "rb")) == NULL) {
    cerr << "Could not open file "<<filename<<" for reading.\n";
    read_image_lock.unlock();
    return false;
  }

  
  //the file is now opened
  
  //first check the eight byte PNG signature;

  if (fread(pbSig, 1, 8, pfFile) != 8) {
    cerr << "Could not read first 8 bytes of "<<filename<<"\n";
    read_image_lock.unlock();
    return false;
  }
 
  
  if(!png_check_sig(pbSig, 8)) {
    cerr << "Magic number of file does not match png\n";
    read_image_lock.unlock();
    return false;
  }

  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,
				   (png_error_ptr)png_cexcept_error, 
				   (png_error_ptr)NULL);
    
  if (!png_ptr) {
    cerr << "Could not create png read struct\n";
    read_image_lock.unlock();
    return false;
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr) {
    cerr << "Could not create png info struct\n";
    png_destroy_read_struct(&png_ptr, NULL, NULL);
    read_image_lock.unlock();
    return false;
  }
  
  cerr << "PNGImage: reading image: " << filename;
  if (flipped_)
    cerr << " (flipped!)";
  cerr << endl;
    
  try {
    
    //initialize the png structure
#if !defined(PNG_NO_STDIO)
    png_init_io(png_ptr, pfFile);
#else
    png_set_read_fn(png_ptr, (png_voidp)pfFile, png_read_data);
#endif

    // Not a .89 version function
    png_set_sig_bytes(png_ptr, 8);
    
    //read all PNG info up to image data
    png_read_info(png_ptr, info_ptr);
    
    //get width, height, bid_depth, Color_type
    png_get_IHDR(png_ptr, info_ptr, &piWidth, 
		 &piHeight, &iBitDepth, &iColorType, NULL,
		 NULL, NULL);
    width_ = (unsigned int)piWidth;
    height_ = (unsigned int)piHeight;
    
    //expand images of all color-type and bit depth to 3x8 bit RGB images
    //let the library process things like alpha, transparance, background
    if(iBitDepth == 16)
      png_set_strip_16(png_ptr);
    if(iColorType == PNG_COLOR_TYPE_PALETTE)
      png_set_expand(png_ptr);
    if (iBitDepth < 8)
      png_set_expand(png_ptr);
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS))
      png_set_expand(png_ptr);
    if (iColorType == PNG_COLOR_TYPE_GRAY ||
	iColorType == PNG_COLOR_TYPE_GRAY_ALPHA)
      png_set_gray_to_rgb(png_ptr);
    
    // set the background color to draw transparent and alpha images over.
    if (png_get_bKGD(png_ptr, info_ptr, &pBackground)) {
      png_set_background(png_ptr, pBackground, PNG_BACKGROUND_GAMMA_FILE, 1, 1.0);
      pBkgColor->red   = (png_byte) pBackground->red;
      pBkgColor->green = (png_byte) pBackground->green;
      pBkgColor->blue  = (png_byte) pBackground->blue;
    } else {
      pBkgColor = NULL;
    }
    
    // after the transformations have been registered update info_ptr data
    png_read_update_info(png_ptr, info_ptr);
    
    
    // get again width, height and the new bit-depth and color-type
    png_get_IHDR(png_ptr, info_ptr, &piWidth, 
		 &piHeight, &iBitDepth,
		 &iColorType, NULL, NULL, NULL);
    
    width_ = (unsigned int)piWidth;
    height_ = (unsigned int)piHeight;

    // row_bytes is the width x number of channels
    ulRowBytes = png_get_rowbytes(png_ptr, info_ptr);
    ulChannels = png_get_channels(png_ptr, info_ptr);
    
    // now we can allocate memory to store the image
    image_.resize((width_)*height_);
    alpha_.resize((width_)*height_);
    
    // now we can allocate memory to store the image
    if (pbImageData) {
      free (pbImageData);
      pbImageData = NULL;
    }

    pbImageData = (png_byte *) malloc(ulRowBytes * (piHeight)
				      * sizeof(png_byte));
    
    if (pbImageData == NULL) {
      png_error(png_ptr, "PNG: out of memory");
    }

    *ppbImageData = pbImageData;

    // and allocate memory for an array of row-pointers

    ppbRowPointers = (png_bytepp) malloc(piHeight * sizeof(png_bytep));
    if (ppbRowPointers == NULL) {
      png_error(png_ptr, "PNG: out of memory");
    }
        
    // set the individual row-pointers to point at the correct offsets
    for (i = 0; i < height_; i++)
      ppbRowPointers[i] = pbImageData + i * ulRowBytes;
    
    // now we can go ahead and just read the whole image
    png_read_image(png_ptr, ppbRowPointers);
    
    // read the additional chunks in the PNG file (not really needed)
    png_read_end(png_ptr, NULL);
    
    for(unsigned v=0; v<height_; v++) {
      for(unsigned u=0; u<width_*ulChannels; u+=ulChannels) {

	int index;
	if (flipped_) {
	  //we have to get the colors from the ppbRowPointers
	  index = (height_-v-1)*(width_)+(u/ulChannels);
	} else {
	  index = v*(width_)+(u/ulChannels);
	}
	image_[index] = Color(ppbRowPointers[v][u]/255.0f,
			      ppbRowPointers[v][u+1]/255.0f,
			      ppbRowPointers[v][u+2]/255.0f);
	alpha_[index] = ppbRowPointers[v][u+3]/255.0f;
      }
    }	  // and we're done
	
    free (ppbRowPointers);
    ppbRowPointers = NULL;
    
  } catch (...) {
    png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    
    if(ppbRowPointers)
      free (ppbRowPointers);
    
    fclose(pfFile);
    
    read_image_lock.unlock();
    return false;
  }
    
  fclose (pfFile);
  valid_ = true;
    
  read_image_lock.unlock();
  return true;
}

#else  // This is if HAVE_PNG is not defined
bool
PNGImage::read_image(const char* filename) {
  cerr << "No png image support.  Not loading file "<<filename<<"\n";
  return false;
}
#endif

bool PNGImage::write_ppm(const char* filename, int bin) {
  ofstream outdata(filename);
  if (!outdata.is_open()) {
    cerr << "PPMImage: ERROR: I/O fault: couldn't write image file: "
	 << filename << "\n";
    return false;
  }
  if (bin)
    outdata << "P6\n# PPM binary image created with rtrt\n";
  else
    outdata << "P3\n# PPM ASCII image created with rtrt\n";
  
  outdata << width_ << " " << height_ << "\n";
  outdata << "255\n";
  
  unsigned char c[3];
  if (bin) {
    for(unsigned v=0;v<height_;++v){
      for(unsigned u=0;u<width_;++u){
	c[0]=(unsigned char)(image_[v*width_+u].red()*255);
	c[1]=(unsigned char)(image_[v*width_+u].green()*255);
	c[2]=(unsigned char)(image_[v*width_+u].blue()*255);
	outdata.write((char *)c, 3);
      }
    }
  } else {
    int count=0;
    for(unsigned v=0;v<height_;++v){
      for(unsigned u=0;u<width_;++u, ++count){
	if (count == 5) { outdata << "\n"; count=0; }
	outdata << (int)(image_[v*width_+u].red()*255) << " ";
	outdata << (int)(image_[v*width_+u].green()*255) << " ";
	outdata << (int)(image_[v*width_+u].blue()*255) << " ";
      }
    }
  }
  return true;
}

const int PNGIMAGE_VERSION = 1;


namespace SCIRun {

  void Pio(SCIRun::Piostream &str, rtrt::PNGImage& obj)
  {
    str.begin_class("PNGImage", PNGIMAGE_VERSION);
    SCIRun::Pio(str, obj.width_);
    SCIRun::Pio(str, obj.height_);
    SCIRun::Pio(str, obj.valid_);
    SCIRun::Pio(str, obj.image_);
    SCIRun::Pio(str, obj.flipped_);
    str.end_class();
  }
} // end namespace SCIRun

/* License for code used to derive png image reading. */

/********************************************************************

This copy of the libpng notices is provided for your convenience.  In case of
any discrepancy between this copy and the notices in the file png.h that is
included in the libpng distribution, the latter shall prevail.

COPYRIGHT NOTICE, DISCLAIMER, and LICENSE:

If you modify libpng you may insert additional notices immediately following
this sentence.

libpng versions 1.0.7, July 1, 2000, through 1.2.5, October 3, 2002, are
Copyright (c) 2000-2002 Glenn Randers-Pehrson
and are distributed according to the same disclaimer and license as libpng-1.0.6
with the following individuals added to the list of Contributing Authors

   Simon-Pierre Cadieux
   Eric S. Raymond
   Gilles Vollant

and with the following additions to the disclaimer:

   There is no warranty against interference with your enjoyment of the
   library or against infringement.  There is no warranty that our
   efforts or the library will fulfill any of your particular purposes
   or needs.  This library is provided with all faults, and the entire
   risk of satisfactory quality, performance, accuracy, and effort is with
   the user.

libpng versions 0.97, January 1998, through 1.0.6, March 20, 2000, are
Copyright (c) 1998, 1999 Glenn Randers-Pehrson, and are
distributed according to the same disclaimer and license as libpng-0.96,
with the following individuals added to the list of Contributing Authors:

   Tom Lane
   Glenn Randers-Pehrson
   Willem van Schaik

libpng versions 0.89, June 1996, through 0.96, May 1997, are
Copyright (c) 1996, 1997 Andreas Dilger
Distributed according to the same disclaimer and license as libpng-0.88,
with the following individuals added to the list of Contributing Authors:

   John Bowler
   Kevin Bracey
   Sam Bushell
   Magnus Holmgren
   Greg Roelofs
   Tom Tanner

libpng versions 0.5, May 1995, through 0.88, January 1996, are
Copyright (c) 1995, 1996 Guy Eric Schalnat, Group 42, Inc.

For the purposes of this copyright and license, "Contributing Authors"
is defined as the following set of individuals:

   Andreas Dilger
   Dave Martindale
   Guy Eric Schalnat
   Paul Schmidt
   Tim Wegner

The PNG Reference Library is supplied "AS IS".  The Contributing Authors
and Group 42, Inc. disclaim all warranties, expressed or implied,
including, without limitation, the warranties of merchantability and of
fitness for any purpose.  The Contributing Authors and Group 42, Inc.
assume no liability for direct, indirect, incidental, special, exemplary,
or consequential damages, which may result from the use of the PNG
Reference Library, even if advised of the possibility of such damage.

Permission is hereby granted to use, copy, modify, and distribute this
source code, or portions hereof, for any purpose, without fee, subject
to the following restrictions:

1. The origin of this source code must not be misrepresented.

2. Altered versions must be plainly marked as such and must not
   be misrepresented as being the original source.

3. This Copyright notice may not be removed or altered from any
   source or altered source distribution.

The Contributing Authors and Group 42, Inc. specifically permit, without
fee, and encourage the use of this source code as a component to
supporting the PNG file format in commercial products.  If you use this
source code in a product, acknowledgment is not required but would be
appreciated.


A "png_get_copyright" function is available, for convenient use in "about"
boxes and the like:

   printf("%s",png_get_copyright(NULL));

Also, the PNG logo (in PNG format, of course) is supplied in the
files "pngbar.png" and "pngbar.jpg (88x31) and "pngnow.png" (98x31).

Libpng is OSI Certified Open Source Software.  OSI Certified Open Source is a
certification mark of the Open Source Initiative.

Glenn Randers-Pehrson
randeg@alum.rpi.edu
October 3, 2002

******************************************************************/
