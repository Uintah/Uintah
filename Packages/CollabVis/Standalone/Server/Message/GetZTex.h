/*
 *
 * SetViewingMethod: Message that encapsulates a change in the viewing
 *                   method (or viewing parameters).
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: May 2001
 *
 */

#ifndef __GET_ZTEX_H_
#define __GET_ZTEX_H_

#include <Message/MessageBase.h>
#include <XML/XMLWriter.h>
#include <XML/XMLReader.h>

namespace SemotusVisum {
namespace Message {

/**************************************
 
CLASS
   GetZTex
   
KEYWORDS
   ZTex, Message
   
DESCRIPTION

   This class provides the infrastructure to create, read, and serialize
   a GetZTex message.
   
****************************************/
class GetZTex : public MessageBase {
public:
  
  //////////
  // Constructor.
  GetZTex();

  //////////
  // Destructor.
  ~GetZTex();

  //////////
  // Finishes serializing the message.
  void finish();

  //////////
  // Sets the new eyepoint params to those supplied.
  inline void setEyePoint( double x, double y, double z ) {
    eyeX = x; eyeY = y; eyeZ = z;
  }

  //////////
  // Fills in the supplied params with the current eyepoint.
  inline void getEyePoint( double &x, double &y, double &z ) const {
    x = eyeX; y = eyeY; z = eyeZ;
  }

  //////////
  // Sets the new look at point params to those supplied.
  inline void setLookAtPoint( double x, double y, double z ) {
    atX = x; atY = y; atZ = z;
  }
  
  //////////
  // Fills in the supplied params with the current look at point.
  inline void getLookAtPoint( double &x, double &y, double &z ) const {
    x = atX; y = atY; z = atZ;
  }

  //////////
  // Sets the new up vector params to those supplied.
  inline void setUpVector( double x, double y, double z ) {
    upX = x; upY = y; upZ = z;
  }

  //////////
  // Fills in the supplied params with the current up vector.
  inline void getUpVector( double &x, double &y, double &z ) const {
    x = upX; y = upY; z = upZ;
  }

  ///////////
  // Returns true if the matrix is set in the message.
  inline bool isMatrixSet() const { return matrixSet; }

  ///////////
  // Sets if the matrix is set in the message.
  inline void setMatrixSet( bool set ) { matrixSet = set; }

  ///////////
  // Returns the matrix from the message. Once this is called, it is the
  // user's responsibility to delete the matrix when finished.
  inline double * getMatrix() { matrixOurs = false; return matrix; }

  ////////////
  // Sets the matrix. 
  void setMatrix( const double * matrix );
  
  //////////
  // Returns a GetZTex message from the given raw data.
  static GetZTex * mkGetZTex( void * data );

protected:
  
  // Eyepoint
  double eyeX, eyeY, eyeZ;

  // Look at point
  double atX, atY, atZ;

  // Up vector
  double upX, upY, upZ;

  // Matrix
  double * matrix;

  // Is matrix set?
  bool matrixSet;

  // Is the matrix our responsibility to delete?
  bool matrixOurs;
};

}
}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 15:46:18  simpson
// Moved CollabVis Server files to Packages/CollabVis/Standalone -- adding these files
//
// Revision 1.1  2003/06/18 21:28:02  simpson
// Adding CollabVis files/dirs
//
// Revision 1.4  2001/08/29 19:58:05  luke
// More work done on ZTex
//
// Revision 1.3  2001/05/14 19:04:52  luke
// Documentation done
//
// Revision 1.2  2001/05/12 02:14:16  luke
// Switched Message base class to MessageBase
//
// Revision 1.1  2001/05/11 20:06:03  luke
// Initial coding of Message abstraction. Documentation not yet done.
//
