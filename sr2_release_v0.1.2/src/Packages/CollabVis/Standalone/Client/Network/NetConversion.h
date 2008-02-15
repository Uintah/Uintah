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

/**
 * Abstraction for network byte order conversion.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class NetConversion {
public:

  /**
   *  Converts floats to network byte order
   *
   * @param input     Input data
   * @param numElems  Number of inputs
   * @param output    If not null, holds output. Else input is converted
   *                  in place.
   */
  static inline void convertFloat( float * input,
				   const int numElems,
				   float * output=NULL );
  /**
   *  Converts ints to network byte order
   *
   * @param input     Input data
   * @param numElems  Number of inputs
   * @param output    If not null, holds output. Else input is converted
   *                  in place.
   */
  static inline void convertInt  ( int   * input,
				   const int numElems,
				   int * output=NULL );
  /**
   *  Converts unsigned ints to network byte order
   *
   * @param input     Input data
   * @param numElems  Number of inputs
   * @param output    If not null, holds output. Else input is converted
   *                  in place.
   */
  static inline void convertUInt ( unsigned * input,
				   const int numElems,
				   unsigned * output=NULL );
  /**
   *  Converts char data interpreted as RGB data. Swaps R and B elements if
   *  necessary.
   *
   * @param input     Input data 
   * @param numElems  Number of inputs (chars)   
   * @param output    If not null, holds output. Else input is converted
   *                  in place.        
   */
  static inline void convertRGB  ( char  * input,
				   const int numElems,
				   char * output=NULL );
  
protected:
  /**
   *  Constructor - This is a fully static class...
   *
   */
  NetConversion() {}
  
  /**
   *  Destructor - This is a fully static class...
   *
   */
  ~NetConversion() {}
  
};

void
NetConversion::convertFloat( float * input, const int numElems,
			     float * output ) {
  if ( output != NULL )
    HomogenousConvertNetworkToHost( (void *)output,
				    (void *)input,
				    FLOAT_TYPE,
				    numElems * sizeof(float) );
  else 
    HomogenousConvertNetworkToHost( (void *)input,
				    (void *)input,
				    FLOAT_TYPE,
				    numElems * sizeof(float) );
}

void
NetConversion::convertInt( int * input, const int numElems,
			   int * output ) {
  if ( output != NULL )
    HomogenousConvertNetworkToHost( (void *)input,
				    (void *)output,
				    INT_TYPE,
				    numElems * sizeof(int) );
  else 
    HomogenousConvertNetworkToHost( (void *)input,
				    (void *)input,
				    INT_TYPE,
				    numElems * sizeof(int) );
}


void 
NetConversion::convertUInt( unsigned * input, const int numElems,
			    unsigned * output ) {
  if ( output != NULL )
    HomogenousConvertNetworkToHost( (void *)input,
				    (void *)output,
				    UNSIGNED_INT_TYPE,
				    numElems * sizeof(unsigned int) );
  else 
    HomogenousConvertNetworkToHost( (void *)input,
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
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:32  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:01:00  simpson
// Adding CollabVis files/dirs
//
// Revision 1.2  2001/10/02 01:52:26  luke
// Fixed xerces problem, compression, other issues
//
// Revision 1.1  2001/08/29 19:56:14  luke
// Added abstraction for net byte order conversion
//
// Revision 1.8  2001/07/16 20:29:36  lu
