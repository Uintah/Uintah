/*
 *
 * NetConversion: Abstraction for network byte order conversion
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: August 2001
 *
 */

#ifndef __NetConversion_h_
#define __NetConversion_h_

#define ASFMT_SHORT_NAMES
#include <Network/AppleSeeds/formatseed.h>

namespace SemotusVisum {
namespace Network {

class NetConversion {
public:

  static inline void convertFloat( float * input,
				   const int numElems,
				   float * output=NULL );
  static inline void convertInt  ( int   * input,
				   const int numElems,
				   int * output=NULL );
  static inline void convertUInt ( unsigned * input,
				   const int numElems,
				   unsigned * output=NULL );
  static inline void convertRGB  ( char  * input,
				   const int numElems,
				   char * output=NULL );
  
protected:
  // This is a fully static class...
  NetConversion() {}
  ~NetConversion() {}
  
};

void
NetConversion::convertFloat( float * input, const int numElems,
			     float * output ) {
  if ( output != NULL )
    HomogenousConvertHostToNetwork( (void *)input,
				    (void *)output,
				    FLOAT_TYPE,
				    numElems * sizeof(float) );
  else 
    HomogenousConvertHostToNetwork( (void *)input,
				    (void *)input,
				    FLOAT_TYPE,
				    numElems * sizeof(float) );
}

void
NetConversion::convertInt( int * input, const int numElems,
			   int * output ) {
  if ( output != NULL )
    HomogenousConvertHostToNetwork( (void *)input,
				    (void *)output,
				    INT_TYPE,
				    numElems * sizeof(int) );
  else 
    HomogenousConvertHostToNetwork( (void *)input,
				    (void *)input,
				    INT_TYPE,
				    numElems * sizeof(int) );
}


void 
NetConversion::convertUInt( unsigned * input, const int numElems,
			    unsigned * output ) {
  if ( output != NULL )
    HomogenousConvertHostToNetwork( (void *)input,
				    (void *)output,
				    UNSIGNED_INT_TYPE,
				    numElems * sizeof(unsigned int) );
  else 
    HomogenousConvertHostToNetwork( (void *)input,
				    (void *)input,
				    UNSIGNED_INT_TYPE,
				    numElems * sizeof(unsigned int) );
}

void
NetConversion::convertRGB( char * input, const int numElems,
			   char * output ) {

  // Do nothing if we don't have to...
  /*  if ( !DifferentOrder() ) {
    std::cerr << " Not converting..." << endl;
    if ( output != NULL ) {
      memcpy( input, output, numElems );
    }
    return; 
    }*/

  // Do nothing if it's not a multiple of 3
  if ( numElems % 3 != 0 ) return; 

  char tmp;

  if ( output != NULL )
    for ( int i = 0; i < numElems; i+= 3 ) {
      output[i]   = input[i+2];
      output[i+1] = input[i+1];
      output[i+2] = input[i];
    }
  else 
    for ( int i = 0; i < numElems; i+= 3 ) {
      tmp = input[i];
      input[i] = input[i+2];
      input[i+2] = tmp;
    }
}

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:25  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:45  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/10/02 01:52:26  luke
// Fixed xerces problem, compression, other issues
//
// Revision 1.1  2001/08/29 19:56:14  luke
// Added abstraction for net byte order conversion
//
// Revision 1.8  2001/07/16 20:29:36  lu
