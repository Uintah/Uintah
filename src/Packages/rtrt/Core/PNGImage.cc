
#include <Packages/rtrt/Core/PNGImage.h>
#include <Packages/rtrt/Core/cexcept.h>
#include <Core/Persistent/PersistentSTL.h>

#include <map>

#ifndef png_jmpbuf
#  define png_jmpbuf(png_otr) ((png_ptr)->jmpbuf)
#endif



using namespace rtrt;
using namespace std;

void
PNGImage::eat_comments_and_whitespace(ifstream &str)
{
  char c;
  str.get(c);
  for(;;) {
    if (c==' '||c=='\t'||c=='\n') {
      str.get(c);
      continue;
    } else if (c=='#') {
      str.get(c);
      while(c!='\n')
        str.get(c);
    } else {
      str.unget();
      break;
    }
  }
}

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

#define PNG_BYTES_TO_CHECK 4
int check_if_png(char *file_name, FILE **fp)
{
   char buf[PNG_BYTES_TO_CHECK];

   /* Open the prospective PNG file. */
   if ((*fp = fopen(file_name, "rb")) == NULL)
      return 0;

   /* Read in some of the signature bytes */
   if (fread(buf, 1, PNG_BYTES_TO_CHECK, *fp) != PNG_BYTES_TO_CHECK)
      return 0;

   /* Compare the first PNG_BYTES_TO_CHECK bytes of the signature.
      Return nonzero (true) if they match */

   return(!png_sig_cmp((png_bytep)buf, (png_size_t)0, PNG_BYTES_TO_CHECK));
}



define_exception_type(const char *);
extern struct exception_context the_exception_context[1];
struct exception_context the_exception_context[1];
png_const_charp msg;

//static OPENFILENAME ofn;

static png_structp png_ptr = NULL;
static png_infop info_ptr = NULL;



//typedef unsigned long png_uint_32;

// cexcept interface

static void
png_cexcept_error(png_structp png_ptr, png_const_charp msg)
{
   if(png_ptr)
     ;
#ifndef PNG_NO_CONSOLE_IO
   fprintf(stderr, "libpng error: %s\n", msg);
#endif
   {
      Throw msg;
   }
}



bool
PNGImage::read_image(const char* filename)
{
  
  static FILE *pfFile = NULL;

  png_byte pbSig[8];
  int iBitDepth;
  int iColorType;
  //  double dGamma;
  //the params
 
  png_uint_32 piWidth, piHeight;
  //  int *piChannels = NULL;
  png_color *pBkgColor = NULL;
  
  png_color_16 *pBackground = NULL;
  png_uint_32 ulChannels;
  png_uint_32 ulRowBytes;
  //image data
  png_byte **ppbImageData = NULL;
  png_byte *pbImageData = *ppbImageData;
  static png_byte **ppbRowPointers = NULL;

  int i;

  //png_structp png_ptr;
  // png_infop info_ptr;

  if((pfFile = fopen(filename, "rb")) == NULL)
    {
      *ppbImageData = pbImageData = NULL;
      return false;
    }

  
  //the file is now opened
  
   //first check the eight byte PNG signature;

  fread(pbSig, 1, 8, pfFile);
 
  
  if(!png_check_sig(pbSig, 8))
    {
 
      *ppbImageData = pbImageData = NULL;
      
      return false;
    }
  
  
 

   png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,
				     (png_error_ptr)png_cexcept_error, 
				     (png_error_ptr)NULL);
    
  

  
  if (!png_ptr)
    {
        *ppbImageData = pbImageData = NULL;
        return false;
    }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr)
    {
      png_destroy_read_struct(&png_ptr, NULL, NULL);
      *ppbImageData = pbImageData = NULL;
        return false;
    }
  
  
    cerr << "PNGImage: reading image: " << filename;
    if (flipped_)
      cerr << " (flipped!)";
    cerr << endl;
    
    
    

    try
      {

	//initialize the png structure
#if !defined(PNG_NO_STDIO)
	png_init_io(png_ptr, pfFile);
#else
	png_set_read_fn(png_ptr, (png_voidp)pfFile, png_read_data);
#endif
	

	
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
        if (png_get_bKGD(png_ptr, info_ptr, &pBackground))
        {
            png_set_background(png_ptr, pBackground, PNG_BACKGROUND_GAMMA_FILE, 1, 1.0);
            pBkgColor->red   = (png_byte) pBackground->red;
            pBkgColor->green = (png_byte) pBackground->green;
            pBkgColor->blue  = (png_byte) pBackground->blue;
        }
        else
        {
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
        
	
	//        *piChannels = ulChannels;
        
	
        // now we can allocate memory to store the image
	
	image_.resize((width_)*height_);
	alpha_.resize((width_)*height_);
	
	
	// now we can allocate memory to store the image
        
        if (pbImageData)
        {
	
	  free (pbImageData);
	  
	  pbImageData = NULL;
        }

	pbImageData = (png_byte *) malloc(ulRowBytes * (piHeight)
					  * sizeof(png_byte));

        if (pbImageData == NULL)
        {
            png_error(png_ptr, "PNG: out of memory");
        }



        *ppbImageData = pbImageData;

        // and allocate memory for an array of row-pointers
        
        if ((ppbRowPointers = (png_bytepp) malloc(piHeight
                            * sizeof(png_bytep))) == NULL)
        {
            png_error(png_ptr, "PNG: out of memory");
        }
        
        // set the individual row-pointers to point at the correct offsets
        
        for (i = 0; i < height_; i++)
            ppbRowPointers[i] = pbImageData + i * ulRowBytes;
        
        // now we can go ahead and just read the whole image
        
        png_read_image(png_ptr, ppbRowPointers);
        
        // read the additional chunks in the PNG file (not really needed)
        
        png_read_end(png_ptr, NULL);
        

	
	for(unsigned v=0;v<height_;v++){
	  for(unsigned u=0;u<width_*ulChannels;u+=ulChannels){
	   
	    if (flipped_) {
	      //we have to get the colors from the ppbRowPointers
	      image_[(height_-v-1)*(width_)+(u/ulChannels)]=rtrt::Color(ppbRowPointers[v][u]/255.0f,
						ppbRowPointers[v][u+1]/255.0f,
						ppbRowPointers[v][u+2]/255.0f);

	      
	      alpha_[(height_-v-1)*(width_)+(u/ulChannels)]=ppbRowPointers[v][u+3]/255.0f;
	
	    } else {
	      image_[v*(width_)+(u/ulChannels)]=rtrt::Color((float)ppbRowPointers[v][u]/255.0f,
						(float)ppbRowPointers[v][u+1]/255.0f,
						(float)ppbRowPointers[v][u+2]/255.0f);
	     


	      alpha_[v*(width_)+(u/ulChannels)]=ppbRowPointers[v][u+3]/255.0f;
	
	    
	    }
	  }
	  
	}	  // and we're done
	
	free (ppbRowPointers);
	ppbRowPointers = NULL;
	
	//  done :)
      }
    
    catch (...)
      {
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	
	*ppbImageData = pbImageData = NULL;
	    
	if(ppbRowPointers)
	  free (ppbRowPointers);
	
	fclose(pfFile);
	    
	return FALSE;
      }
    
    fclose (pfFile);
    valid_ = true;
    
    return true;
}
    
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
