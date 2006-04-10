/**********************************************************************/
/* JPEG.h: This file contains the class declarations for JPEG data
           compression.
		   
   Author:        Eric Luke
   Date:          May 20, 2000
*/   
/*********************************************************************/


#ifndef _JPEG_H_
#define _JPEG_H_

#include <Compression/Compression.h>

#ifdef __sgi
#pragma set woff 3303
#endif

#include <X11/Xmd.h>

extern "C" {
#include <jpeglib.h>
}

namespace SemotusVisum {
namespace Compression {

//using namespace std;

class JPEGCompress : public Compressor {
public:
  JPEGCompress();
  JPEGCompress(int quality);
  virtual ~JPEGCompress();
  
  /* Compress the data. Note that the routine will allocate space for the
     output buffer if the pointer points to NULL.
     Returns : Length of the compressed data or -1 on error. */
  virtual int compress(DATA * input,   // Input buffer
		       int width,      // Width of buffer
		       int height,     // Height of buffer   
		       DATA ** output, // Pointer to output buffer
		       int bps=3,      // Bytes per sample
		       int delta=1);   // Input delta. 
		       
  /* Decompress the data. Note that the routine will allocate space for the
     output buffer if the pointer points to NULL.
     Returns : Length of the decompressed data or -1 on error. */
  virtual int decompress(DATA * input,      // Input buffer
		         int buffer_length, // Input buffer length
		         DATA ** output,    // Pointer to output buffer
			 int delta=1);      // Output delta.

  /* Sets the quality of the image. Ranges from 0-100, but only 5-95 is
     useful. Higher values represent higher quality. */
  virtual inline void Quality(int newquality)
    {
      if (newquality > 100) newquality = 100;
      if (newquality < 0)   newquality = 0;
      jpeg_set_quality(&ccinfo, newquality, false);
    }

  ////////////
  // Returns the name of the compressor
  virtual inline const char * const getName() const { return name;  }

  ////////////
  // Returns true if the compressor is lossy.
  virtual inline bool               isLossy() const { return lossy; }

  ////////////
  // Returns true if the compressor needs RGB conversion, or false
  // if the compressor handles that itself.
  virtual inline bool               needsRGBConvert() const { return convert; }

  /////////////
  // Static version to get our name.
  static const char * const Name() { return name; }

protected:
  static const char * const name;

  static const bool lossy = true;
  static const bool convert = false;
  
  struct jpeg_decompress_struct dcinfo;
  struct jpeg_compress_struct ccinfo;
  struct jpeg_error_mgr jerr;

  void init();
};



/* A source input manager for the jpeg decompression library. It
   requires that we already have the image available in a buffer. */
typedef struct {
  struct jpeg_source_mgr pub;
  
  JOCTET * buffer;
  int buffer_size;
} sourceManager;

typedef sourceManager * sm_ptr;

/* A destination output manager for the jpeg compression library. It
   requires that we already have the image available in a buffer. */
typedef struct {
  struct jpeg_destination_mgr pub;
  
  JOCTET * buffer;
  int bytes_written;
} destManager;

typedef destManager * dm_ptr;

/* Utility functions for the JPEG library */
  void init_source (j_decompress_ptr cinfo);
  boolean fill_input_buffer(j_decompress_ptr cinfo);
  void skip_input_data(j_decompress_ptr cinfo, long num_bytes);
  void term_source(j_decompress_ptr cinfo);
  void jpeg_buffer_src(j_decompress_ptr cinfo, JSAMPLE * buffer,
		       int buffer_size);

  void init_destination (j_compress_ptr cinfo);
  boolean empty_output_buffer(j_compress_ptr cinfo);
  void term_destination(j_compress_ptr cinfo);
  void jpeg_buffer_dest(j_compress_ptr cinfo, JSAMPLE * buffer);



} // End namespace Compression
}

#endif // _JPEG_H_
