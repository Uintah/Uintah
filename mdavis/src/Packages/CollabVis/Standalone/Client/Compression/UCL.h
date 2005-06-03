#ifndef _UCL_H_
#define _UCL_H_

#include <Compression/Compression.h>
#include <Compression/ucl/ucl.h>

namespace SemotusVisum {
namespace Compression {
/**
 * This class provides access to the UCL compression library
 *
 * @author  Eric Luke
 * @version $Revision$
 */
 
class UCLCompress : public Compressor {
public:
  UCLCompress();
  UCLCompress(int level);
  virtual ~UCLCompress();

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

  /* Change the compression level of the compressor. Values can range from
     1 (fastest, least compression) to 10 (slowest, most compression). */
  virtual inline void CompressionLevel(int newlevel)
    {
      compressionLevel = newlevel;
      if (compressionLevel > 10) compressionLevel = 10;
      if (compressionLevel < 1)  compressionLevel = 1;
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
  static const bool lossy = false;
  static const bool convert = true;
  

  int compressionLevel; // Compression level (1-10)
};

} // End namespace Compression
} 
#endif // _UCL_H_
 
