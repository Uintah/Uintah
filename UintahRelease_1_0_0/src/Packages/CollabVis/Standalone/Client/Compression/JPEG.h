

#ifndef _JPEG_H_
#define _JPEG_H_

#include <Compression/Compression.h>

#ifdef __sgi
#pragma set woff 3303
#endif

#include <X11/Xmd.h>

extern "C" {
#ifdef EXTERN
#undef EXTERN
#endif
#include <jpeglib.h>
#undef EXTERN
}

namespace SemotusVisum {

/**
 * This class provides access to the JPEG compression library
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class JPEGCompress : public Compressor {
public:
  /**
   *  Constructor.
   *
   */
  JPEGCompress();
  
  /**
   *  Constructor that explicitly sets compression quality.
   *
   * @param quality        Quality to set.
   */
  JPEGCompress(int quality);
  
  /**
   *  Destructor.
   *
   */
  virtual ~JPEGCompress();
  

  /**
   * Compress the data. Note that the routine will allocate space for the
   *  output buffer if the pointer points to NULL.
   *
   * @param input               Input buffer
   * @param width               Width of buffer
   * @param height              Height of buffer           
   * @param output              Pointer to output buffer
   * @param bps                 Bytes per sample 
   * @param delta               Input delta
   * @return                    Length of compressed data or -1 on error.
   */
  virtual int compress(DATA * input, int width, int height,
		       DATA ** output, int bps=3, int delta=1);
  
  
  /**
   * Decompress the data. Note that the routine will allocate space for the
   * output buffer if the pointer points to NULL.
   *
   * @param input              Input buffer
   * @param buffer_length      Input buffer length
   * @param output             Pointer to output buffer
   * @param delta              Output delta.
   * @return                   Length of the decompressed data or -1 on
   *                           error. 
   */
  virtual int decompress(DATA * input,     
		         int buffer_length, 
		         DATA ** output,   
			 int delta=1);      

  
  /**
   * Returns the name of the compressor
   *
   * @return    Name of the compressor.
   */
  virtual inline const char * const getName() const { return name;  }

  /** 
   * Returns true if the compressor is lossy.
   *
   * @return    True if the compressor is lossy; else false.
   */
  virtual inline bool               isLossy() const { return lossy; }

  /**
   * Returns true if the compressor needs RGB conversion, or false
   * if the compressor handles that itself.
   *
   * @return    Returns true if the compressor needs RGB conversion, or false
   *            if the compressor handles that itself.
   */
  virtual inline bool               needsRGBConvert() const { return convert; }

  /** 
   * Static method to get our name.
   *
   * @return     Name of this compressor.
   */
  static const char * const Name() { return name; }
  
  /**
   * Sets the quality of the image. Ranges from 0-100, but only 5-95 is
   *  useful. Higher values represent higher quality. 
   *
   * @param newquality      Quality of the image (0-100)
   */
  virtual inline void Quality(int newquality)
    {
      if (newquality > 100) newquality = 100;
      if (newquality < 0)   newquality = 0;
      jpeg_set_quality(&ccinfo, newquality, false);
    }


protected:
  
  /** Name of this compressor. */
  static const char * const name;
  
  /** Is this a lossy compressor? */
  static const bool lossy = true;

  /** Do we need to do RGB conversion? */
  static const bool convert = false;

  /** JPEG decompress struct */
  struct jpeg_decompress_struct dcinfo;

  /** JPEG compress struct */
  struct jpeg_compress_struct ccinfo;

  /** JPEG error manager */
  struct jpeg_error_mgr jerr;

  /** Initializes the compressor */
  void init();
};



/** A source input manager for the jpeg decompression library. It
    requires that we already have the image available in a buffer. */
typedef struct {
  struct jpeg_source_mgr pub;
  
  JOCTET * buffer;
  int buffer_size;
} sourceManager;

/** Pointer to source manager */
typedef sourceManager * sm_ptr;

/** A destination output manager for the jpeg compression library. It
    requires that we already have the image available in a buffer. */
typedef struct {
  struct jpeg_destination_mgr pub;
  
  JOCTET * buffer;
  int bytes_written;
} destManager;

/** Pointer to dest manager */
typedef destManager * dm_ptr;


/** Initializes source */
void init_source (j_decompress_ptr cinfo);

/** Fills input buffer */
boolean fill_input_buffer(j_decompress_ptr cinfo);

/** Skips num_bytes into the input data */
void skip_input_data(j_decompress_ptr cinfo, long num_bytes);

/** Terminates the input source */
void term_source(j_decompress_ptr cinfo);

/** Sets the source jpeg buffer */
void jpeg_buffer_src(j_decompress_ptr cinfo, JSAMPLE * buffer,
		     int buffer_size);

/** Initializes the output buffer */
void init_destination (j_compress_ptr cinfo);

/** Returns true if the output buffer is empty */
boolean empty_output_buffer(j_compress_ptr cinfo);

/** Terminates the destination */
void term_destination(j_compress_ptr cinfo);

/** Sets the destination jpeg buffer */
void jpeg_buffer_dest(j_compress_ptr cinfo, JSAMPLE * buffer);



}

#endif // _JPEG_H_
