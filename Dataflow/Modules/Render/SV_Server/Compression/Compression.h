/**********************************************************************/
/* Compression.h: This file contains the virtual superclass for
                  data compression.
		   
   Author:        Eric Luke
   Date:          May 20, 2000
*/   
/*********************************************************************/
    							      
#ifndef _COMPRESSION_H_
#define _COMPRESSION_H_

typedef unsigned char DATA;
#include <stdio.h>
#include <Malloc/Allocator.h>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {
namespace Compression {

class Compressor {
public:
  Compressor() {}
  virtual ~Compressor() {}

  /* Compress the data. Note that the routine will allocate space for the
     output buffer if the pointer points to NULL.
     Returns : Length of the compressed data or -1 on error. */
  virtual int compress(DATA * input,      // Input buffer
		       int width,         // Width of buffer
		       int height,        // Height of buffer   
		       DATA ** output,    // Pointer to output buffer
		       int bps=3,         // Bytes per sample
		       int delta=1) = 0;  // Input delta. 
		       
  /* Decompress the data. Note that the routine will allocate space for the
     output buffer if the pointer points to NULL.
     Returns : Length of the decompressed data or -1 on error. */
  virtual int decompress(DATA * input,      // Input buffer
		         int buffer_length, // Input buffer length
		         DATA ** output,    // Pointer to output buffer
			 int delta=1) = 0;  // Output delta.

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
  
protected:
  static const char * const name;
  
  static const bool lossy = false;

  static const bool convert = true;
  
  inline DATA * allocateMemory(unsigned nbytes)
  {
    DATA * tmp;

    if (!(tmp = scinew DATA[nbytes])) {
      char buffer[100];
      snprintf( buffer, 100, "Unable to allocate %d bytes", nbytes );
      perror(buffer/*"Unable to allocate memory"*/);
    }

    return tmp;
  }
  
};

} // End namespace Compressor
}
#endif // _COMPRESSION_H_
