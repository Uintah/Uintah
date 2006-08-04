/*
 *
 * BBox: Utility for 3D bounding box.
 *
 * $Id$
 *
 * Written by:
 *   Author: Eric Luke
 *   Department of Computer Science
 *   University of Utah
 *   Date: June 2001
 *
 */


#ifndef __BBOX_H_
#define __BBOX_H_

#include <values.h>
#include <Util/stringUtil.h>
#include <Util/Point.h>
#include <Util/MinMax.h>

namespace SemotusVisum {

/**
 * Enapsulates the concept of a 3D bounding box.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class BBox {
public:
  /**
   * Constructor - sets an empty box
   *
   */
  BBox() : _valid( false ) {}
  
  /**
   * Destructor
   *
   */
  ~BBox() {}

  /**
   * Returns true if the box is valid (of non-zero size)
   *
   * @return True if the box is valid
   */
  inline bool valid() const { return _valid; }
  
  /**
   * Returns the minimum point of the box
   *
   * @return Minimum corner of box
   */
  inline Point3d min() const { return _min; }
  
  /**
   * Returns the maximum point of the box
   *
   * @return Maximum corner of box
   */
  inline Point3d max() const { return _max; }
  
  /**
   * Returns the spatial center of the box
   *
   * @return Center of box
   */
  inline Point3d middle() const {
    return Point3d( (_max.x+_min.x)/2.0f,
		    (_max.y+_min.y)/2.0f,
		    (_max.z+_min.z)/2.0f );
  }
  
  /**
   * Extends the box to encompass the given point
   *
   * @param p     Point to place in box
   */
  inline void extend( const Point3d &p ) {
    if ( valid() ) {
      _min = Min( p, _min );
      _max = Max( p, _max );
    }
    else {
      _min = p;
      _max = p;
      _valid = true;
    }
  }
  
  /**
   * Extends the box to encompass the given box
   *
   * @param b     Box to extend this box with
   */
  inline void extend( const BBox &b ) {
    if ( b.valid() ) {
      extend( b.min() );
      extend( b.max() );
    }
  }
  
  /**
   * Returns a string rep of the box
   *
   * @return String rep of the box.
   */
  inline string toString() {
    if ( !_valid ) return "Invalid BBox";
    return string( "[" + _min.toString() + " -> " + _max.toString() + "]" );
  }
  
protected:

  /// Box corner
  Point3d _min, _max;

  /// Is the box valid?
  bool _valid;

  
};


}

#endif
