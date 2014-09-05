#ifndef _RLE_H_
#define _RLE_H_

#include <stdlib.h>
#include <stdio.h>
#include <Compression/Compression.h>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {

/** Max run length for the encoding */
const int MAX_RUN_LENGTH = 255;

/** Run code for the encoding */
const int RUN_CODE = 0;

/** Replace code for the encoding */
const int CODE_REPLACE = 1;

/**
 * This class provides access to the Run Length Encoding data compression.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class RLECompress : public Compressor {
public:
  /**
   *  Constructor.
   *
   */
  RLECompress();
  
  /**
   *  Destructor.
   *
   */
  virtual ~RLECompress();

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

  
 protected:
  /** Name of this compressor. */
  static const char * const name;
  
  /** Is this a lossy compressor? */
  static const bool lossy = true;

  /** Do we need to do RGB conversion? */
  static const bool convert = true;
  
  /** Encodes the data of length input_len, and places it into output.
   *  Returns the number of output bytes */
  int encode(DATA * input, int input_len, DATA * output);

  /** Decodes the data of length input len, and places it into output.
   *  Returns the number of output bytes */
  int decode(DATA * input, int input_len, DATA * output);

  
  /** maximum length for runs or sequences    */
  static const int MAX_LEN = (0x7f);

  /** bit 7 == 1 : run follows
   * bit 6 - 0  : length of run */
  static const int MAX_RUN_HEADER = (0xff);
  
  /** bit 7 == 0 : unencode sequence follows */
  static const int MAX_SEQ_HEADER=(0x7f);
  
  /** bit 6 - 0  : length of sequence
   * bit 7 == 1 : run follows */
  static const int RUN = (0x80);

  /** bit 7 == 0 : unencoded sequence follows */
  static const int SEQ = (0x00);
};



}

#endif // _RLE_H_
