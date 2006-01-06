#ifndef _ZLIB_H_
#define _ZLIB_H_

#include <Compression/Compression.h>

#ifdef __sgi
#pragma set woff 3303
#endif

namespace SemotusVisum {

/**
 * This class provides access to the ZLIB compression library
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class ZLIBCompress : public Compressor {
public:
  /**
   *  Constructor.
   *
   */
  ZLIBCompress();

  
  /**
   *  Destructor.
   *
   */
  virtual ~ZLIBCompress();

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
  static const bool lossy = false;

  /** Do we need to do RGB conversion? */
  static const bool convert = true;
};




}
#endif // _ZLIB_H_
//
// $Log$
// Revision 1.1  2003/07/22 20:59:03  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 20:17:59  simpson
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
