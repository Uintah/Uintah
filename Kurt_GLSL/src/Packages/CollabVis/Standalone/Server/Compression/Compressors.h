#ifndef __Compressors_h_
#define __Compressors_h_

#include <string.h>

#include <Compression/JPEG.h>
#include <Compression/LZO.h>
#include <Compression/RLE.h>
#include <Compression/ZLIB.h>
//#include <Compression/UCL.h>

namespace SemotusVisum {
namespace Compression {

enum {
  JPEG,
  LZO,
  RLE,
  ZLIB,
  NONE,
  CERROR
  //,UCL
};

const int NUM_COMPRESSORS = (int)NONE;

///////////////
// Returns a compressor based on the given name. Also sets 'type' to the
// appropriate enumerated type if a compressor is found.
inline Compressor * mkCompressor( const char * name, int& type ) {
  // Initial setting.
  type = CERROR;
  
  if ( !name ) return NULL;
  
  if ( !strcasecmp( name, JPEGCompress::Name() ) ) {
    type = JPEG;
    return new JPEGCompress;
  }
  else if ( !strcasecmp( name, LZOCompress::Name() ) ) {
    type = LZO;
    return new LZOCompress;
  }
  else if ( !strcasecmp( name, RLECompress::Name() ) ) {
    type = RLE;
    return new RLECompress;
  }
  else if ( !strcasecmp( name, ZLIBCompress::Name() ) ) {
    type = ZLIB;
    return new ZLIBCompress;
  }
  else if ( !strcasecmp( name, "None" ) ) {
    type = NONE;
    return NULL;
  }
  else 
    return NULL;
}

}
}
#endif
