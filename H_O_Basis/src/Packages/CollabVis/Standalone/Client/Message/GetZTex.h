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

namespace SemotusVisum {

/**
 * This class provides the infrastructure to create, read, and serialize
 * a GetZTex message.
 *
 * @author  Eric Luke
 * @version $Revision$
 */

class GetZTex : public MessageBase {
public:
  
  /**
   *  Constructor.
   *
   */
  GetZTex();

  /**
   * Destructor.
   *
   */
  ~GetZTex();

  /**
   * Finishes serializing the message.
   *
   */
  void finish();

  /**
   *  Sets the new eyepoint params to those supplied.
   *
   * @param x     X coordinate
   * @param y     Y coordinate     
   * @param z     Z coordinate     
   */
  inline void setEyePoint( const double x, const double y, const double z ) {
    eyeSet = true;
    eyeX = x; eyeY = y; eyeZ = z;
  }

  /**
   *  Fills in the supplied params with the current eyepoint.
   *
   * @param x     X coordinate
   * @param y     Y coordinate     
   * @param z     Z coordinate   
   */
  inline void getEyePoint( double &x, double &y, double &z ) const {
    x = eyeX; y = eyeY; z = eyeZ;
  }

  /**
   *  Sets the new look at point params to those supplied.
   *
   * @param x     X coordinate
   * @param y     Y coordinate     
   * @param z     Z coordinate   
   */
  inline void setAtPoint( const double x, const double y, const double z ) {
    atSet = true;
    atX = x; atY = y; atZ = z;
  }
  
  /**
   *  Fills in the supplied params with the current look at point.
   *
   * @param x     X coordinate
   * @param y     Y coordinate     
   * @param z     Z coordinate   
   */
  inline void getAtPoint( double &x, double &y, double &z ) const {
    x = atX; y = atY; z = atZ;
  }

  /**
   *  Sets the new up vector params to those supplied.
   *
   * @param x     X component
   * @param y     Y component
   * @param z     Z component
   */
  inline void setUpVector( const double x, const double y, const double z ) {
    upSet = true;
    upX = x; upY = y; upZ = z;
  }

  /**
   *  Fills in the supplied params with the current up vector.
   *
   * @param x     X component
   * @param y     Y component
   * @param z     Z component
   */
  inline void getUpVector( double &x, double &y, double &z ) const {
    x = upX; y = upY; z = upZ;
  }

  /**
   *  Returns true if the matrix is set in the message.
   *
   * @return True if the view matrix is set in the message
   */
  inline bool isMatrixSet() const { return matrixSet; }

  
  /**
   *  Sets if the matrix is set in the message.
   *
   * @param set   True if the message contains a matrix.
   */
  inline void setMatrixSet( const bool set ) { matrixSet = set; }

  /**
   *  Returns the matrix from the message. Once this is called, it is the
   * user's responsibility to delete the matrix when finished.
   *
   * @return Viewing matrix, or NULL if not set.
   */
  inline double * getTransform() { matrixOurs = false; return matrix; }

  /**
   *  Sets the matrix. 
   *
   * @param matrix        Viewing matrix
   */
  void setTransform( const double * matrix );
  
  /**
   *  Returns a GetZTex message from the given raw data.
   *
   * @param data  Raw Data
   * @return      New message, or NULL on error.
   */
  static GetZTex * mkGetZTex( void * data );

protected:
  
  /** Eye coordinate */
  double eyeX, eyeY, eyeZ;

  /** Look at coordinate */
  double atX, atY, atZ;

  /** Up vector */
  double upX, upY, upZ;

  /** View Matrix */
  double * matrix;

  /** Is matrix set? */
  bool matrixSet;

  /** Is parameter set? */
  bool eyeSet, atSet, upSet;

  /** Is the matrix our responsibility to delete? */
  bool matrixOurs;
};

}
#endif
//
// $Log$
// Revision 1.1  2003/07/22 20:59:26  simpson
// Adding CollabVis Client directory to Packages/CollabVis/Standalone
//
// Revision 1.1  2003/06/18 21:00:10  simpson
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
