/*
 *
 * Color: Utility for capturing notion of color.
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

#ifndef __COLOR_H_
#define __COLOR_H_

#include <Util/stringUtil.h>

namespace SemotusVisum {

/**
 * Utility for capturing notion of color.
 *
 * @author  Eric Luke
 * @version $Revision$
 */
class Color {
public:
  /**
   *  Constructor that sets values
   *
   * @param r     Red component (0-255)
   * @param g     Green component (0-255)    
   * @param b     Blue component (0-255)    
   */
  Color( const int r, const int g, const int b ) : r(r), g(g), b(b) {
    _array[0] = r/255.0; _array[1] = g/255.0; _array[2]=b/255.0;
    _array[3] = 1.0;
  }
  
  /**
   * Copy constructor
   *
   * @param c    Color to copy
   */
  Color( const Color &c ) : r(c.r), g(c.g), b(c.b) {
    _array[0] = r/255.0; _array[1] = g/255.0; _array[2]=b/255.0;
    _array[3] = 1.0;
  }
  
  /**
   * Constructor - sets default values (NOT 0,0,0 )
   *
   */
  Color() : r(-1), g(-1), b(-1) {}
  
  /**
   * Destructor
   *
   */
  ~Color() {}

  /**
   * Returns the red component (0-255)
   *
   * @return Red component
   */
  inline int getRed() const { return r; }
  
  /**
   * Returns the green component (0-255)
   *
   * @return Green component
   */
  inline int getGreen() const { return g; }
  
  /**
   * Returns the blue component (0-255)
   *
   * @return Blue component
   */
  inline int getBlue() const { return b; }

  /**
   * Returns true if the color has been set (represents an actual color).
   *
   * @return True if set, false otherwise.
   */
  inline bool set() const { return r > 0 && g > 0 && b > 0; }
  
  /**
   * Returns a string rep of the color
   *
   * @return String rep of the color
   */
  inline string toString() const {
    return string("[") + mkString(r) + ", " + mkString(g) +
      ", " + mkString(b) + "]";
  }
  
  /**
   * Returns the color as a floating point (0-1.0) array
   *
   * @return Color as an array
   */
  inline float* array() { return _array; }
  
protected:
  /// Color component
  int r, g, b;

  /// Array
  float _array[4];
};

}
#endif
