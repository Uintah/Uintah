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

#include <Compression/ZLIB.h>
#include <zlib.h>
#include <stdlib.h>
#include <string.h>

namespace SemotusVisum {

const char * const
ZLIBCompress::name = "zip";

ZLIBCompress::ZLIBCompress() {

}

ZLIBCompress::~ZLIBCompress() {

}

int
ZLIBCompress::compress(DATA * input, int width, int height, DATA ** output,
		       int bps, int delta) {

  DATA *tmp = NULL;
  uLongf destLen = 0;
  unsigned bufferSpace = (unsigned)( (width * height * bps) * 1.2 + 12 );
  destLen = bufferSpace;
  
  if ( !output ) // We need this, at least
    return -1;

  tmp = new DATA[ bufferSpace ]; // Says zlib.h

  // Compress the data.
  int result;
  if ( (result=::compress( tmp, &destLen, input, width*height*bps ))
       != Z_OK ) {
    delete tmp;
    if ( result > 0 ) result = -result;
    return result;
  }

  /* Allocate output space if needed, and copy data */
  if ( !output[0] )
    *output = new DATA[ destLen ];
  memcpy( *output, tmp, destLen );

  // Clean up and return.
  delete tmp;
  return (int)destLen;
}


int
ZLIBCompress::decompress(DATA * input, int buffer_length, DATA ** output, 
			 int delta) {
#if 1
  DATA *tmp;
  int sizeFactor = 10; // Multiples of the input length
  uLongf destLen = sizeFactor * buffer_length;
  
  if ( !output ) // We need this, at least
    return -1;

  tmp = new DATA[ destLen ]; // should be enough...
  
  // Decompress the data.
  int result;

  do {
    result = ::uncompress( tmp, &destLen, input, buffer_length );
    if ( result == Z_BUF_ERROR || result == Z_MEM_ERROR ) {
      delete tmp;
      sizeFactor += 5;
      destLen = sizeFactor * buffer_length;
      tmp = new DATA[ destLen ];
    }
    else if ( result != Z_OK ) {
      delete tmp;
      return -1;
    }
  } while ( result != Z_OK );

  /* Allocate output space if needed, and copy data */
  if ( !output[0] )
    *output = new DATA[ destLen ];
  memcpy( *output, tmp, destLen );

  // Clean up and return.
  delete tmp;
  return (int)destLen;
  
#else
  DATA *tmp = NULL;
  z_stream inStream;
  bool allocated = false;
  unsigned bufferSize = buffer_length * 2;
  
  if ( !output ) // We need this, at least
    return -1;

  /* We don't know how big our decompressed data will be. Thus, we have
     to decompress things stream-wise. */

  // Allocate memory
  tmp = new DATA[ bufferSize ]; // A reasonable start
  if ( !output[0] ) {
    *output = new DATA[ bufferSize ];
    allocated = true;
  }
  else
    return -1; // We just need to allocate our own memory.
  
  // Set up stream parameters
  inStream.next_in  = input;
  inStream.avail_in = bufferSize;
  inStream.total_in = 0;

  inStream.next_out  = tmp;
  inStream.avail_out = bufferSize;
  inStream.total_out = 0;

  inStream.zalloc = Z_NULL;
  inStream.zfree  = Z_NULL;
  

  // Decompress away
  int result;
  DATA* currentOutputPosition = *output;
  unsigned outputLength = 0;
  DATA * tmpRealloc;
  
  int oldTotal = 0;
  
  do  {
    result = inflate( &inStream, Z_SYNC_FLUSH );
    if ( result == Z_DATA_ERROR || result == Z_STREAM_ERROR ||
	 result == Z_MEM_ERROR  || result == Z_BUF_ERROR ) {
      // We're hosed. Do no error recovery; just say goodbye.

      inflateEnd( &inStream ); // Deallocates internal zlib memory.
      delete tmp;
      if ( allocated ) delete output;
      return -1;
    }

    if ( result == Z_OK ) {
      
      // If we haven't filled the output buffer yet, call it again
      if ( inStream.avail_out != 0 ) continue;

      // Otherwise, we've filled the output buffer.
      
      // Allocate more room in the output buffer
      tmpRealloc = (DATA *)::realloc( *output, bufferSize );
      if ( tmpRealloc != NULL ) {
	*output = tmpRealloc;
	currentOutputPosition = *output + outputLength;
      }
      else {
	// Crap - malloc failed. Let's puke.
	inflateEnd( &inStream ); // Deallocates internal zlib memory.
	delete tmp;
	if ( allocated ) delete output;
	return -1;
      }
      
      // Copy the temp buffer to the output buffer
      memcpy( currentOutputPosition, tmp, bufferSize );
      currentOutputPosition += bufferSize;
      outputLength += bufferSize;
      
      // Update the avail_out parameter in inStream.
      inStream.avail_out = bufferSize;
      inStream.next_out = tmp;
    }
    
  } while ( result != Z_STREAM_END );

  // Do the final allocate/copy
  unsigned bytes = bufferSize-inStream.avail_out;
  // Allocate more room in the output buffer
  tmpRealloc = (DATA *)::realloc( *output, bytes );
  if ( tmpRealloc != NULL ) {
    *output = tmpRealloc;
    currentOutputPosition = *output + outputLength;
  }
  else {
    // Crap - malloc failed. Let's puke.
    inflateEnd( &inStream ); // Deallocates internal zlib memory.
    delete tmp;
    if ( allocated ) delete output;
    return -1;
  }
  
  // Copy the temp buffer to the output buffer
  memcpy( currentOutputPosition, tmp, bytes );
  outputLength += bytes;
  
  // Now, outputLength should equal inStream.total_out;
  if ( outputLength != inStream.total_out ) {
    return -1;
  }

  delete tmp;
  return outputLength;
#endif
}


}
//
// $Log$
// Revision 1.1  2003/07/22 20:59:03  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 20:17:59  simpson
// Adding CollabVis files/dirs
//
// Revision 2.3  2001/08/01 00:16:05  luke
// Compiles on SGI. Fixed list allocation bug in NetDispatchManager
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
