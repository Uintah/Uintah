/*
 *
 * Zlib: Provides access to the ZLIB compression library
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */


#ifndef _ZLIB_H_
#define _ZLIB_H_

#include <Compression/Compression.h>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {
namespace Compression {

class ZLIBCompress : public Compressor {
public:
  ZLIBCompress();
  virtual ~ZLIBCompress();

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
  

  static const bool lossy = false;
  static const bool convert = true;
};




} // End namespace Compression

}
#endif // _ZLIB_H_
//
// $Log$
// Revision 1.1  2003/07/22 15:45:58  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:19:48  simpson
// Adding CollabVis files/dirs
//
// Revision 2.4  2001/10/08 17:30:02  luke
// Added needsRGBConvert to compressors so we can avoid RGB conversion where unnecessary
//
// Revision 2.3  2001/07/16 20:27:33  luke
// Updated compression libraries...
//
// Revision 2.2  2001/06/13 21:25:59  luke
// Added bits per sample field into the compress() functions
//
// Revision 2.1  2001/06/12 21:26:08  luke
// Added zlib support
//
// Revision 2.0  2001/06/12 21:25:56  luke
// Added zlib support
//
