/**********************************************************************/
/* RLE.h: This file contains the class declarations for Run Length
          Encoding data compression.
		   
   Author:        Eric Luke
   Date:          May 20, 2000
*/   
/*********************************************************************/


#ifndef _RLE_H_
#define _RLE_H_

#include <stdlib.h>
#include <stdio.h>
#include <Compression/Compression.h>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {
namespace Compression {
    
const int MAX_RUN_LENGTH = 255;
const int RUN_CODE = 0;
const int CODE_REPLACE = 1;

class RLECompress : public Compressor {
public:
  RLECompress();
  virtual ~RLECompress();

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
  static const bool convert = true;
  int encode(DATA * input, int input_len, DATA * output);
  int decode(DATA * input, int input_len, DATA * output);

  
  /* maximum length for runs or sequences    */
  static const int MAX_LEN = (0x7f);

  /* bit 7 == 1 : run follows */
  /* bit 6 - 0  : length of run */
  static const int MAX_RUN_HEADER = (0xff);
  
  /* bit 7 == 0 : unencode sequence follows */
  static const int MAX_SEQ_HEADER=(0x7f);
  
  /* bit 6 - 0  : length of sequence */
  /* bit 7 == 1 : run follows */
  static const int RUN = (0x80);

  /* bit 7 == 0 : unencoded sequence follows */
  static const int SEQ = (0x00);
};



} // End namespace Compression
}

#endif // _RLE_H_
