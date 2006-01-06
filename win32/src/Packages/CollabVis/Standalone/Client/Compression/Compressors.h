#ifndef __Compressors_h_
#define __Compressors_h_

#include <Util/stringUtil.h>

#include <Compression/JPEG.h>
#include <Compression/LZO.h>
#include <Compression/RLE.h>
#include <Compression/ZLIB.h>
//#include <Compression/UCL.h>

namespace SemotusVisum {

enum {
  JPEG, /** JPEG compression */
  LZO,  /** LZO compression */
  RLE,  /** RLE compression */
  //ZLIB,
  NONE, /** No compression */
  CERROR /** Compression error */
  //,UCL
};


/** Compression methods */
static const char * const compressionMethods[] = { JPEGCompress::Name(), 
						   LZOCompress::Name(),
						   RLECompress::Name(),
						   "None",
						   0 };

/** Number of compressors */
const int NUM_COMPRESSORS = (int)NONE;

/**
 * Returns a compressor based on the given name. Also sets 'type' to the
 * appropriate enumerated type if a compressor is found. 
 *
 * @param name            Name of the compressor
 * @param type            Type of the compressor (set)
 * @return                Pointer to new compressor
 */
inline Compressor * mkCompressor( const string &name, int& type ) {
  // Initial setting.
  type = CERROR;
  
  if ( name.empty() ) return NULL;
  
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
  //else if ( !strcasecmp( name, ZLIBCompress::Name() ) ) {
  //  type = ZLIB;
  //  return new ZLIBCompress;
  //}
  else if ( !strcasecmp( name, "None" ) ) {
    type = NONE;
    return NULL;
  }
  else 
    return NULL;
}

/**
 * Returns a compressor based on the given type of compressor.
 *
 * @param name        Numeric type of the compressor
 * @param type        Type of the compressor (set)
 * @return            Pointer to new compressor
 */
inline Compressor *mkCompressor( const int name, int &type ) {
  return mkCompressor( string(compressionMethods[ name ]), type );
}

}
#endif
