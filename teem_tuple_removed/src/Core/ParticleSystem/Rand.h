#ifndef __RAND_H__
#define __RAND_H__

#if defined __cplusplus

#ifndef __VECTOR_H__
#include <Vector.h>
#endif

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#define DIVRAND         ( 1.0 / RAND_MAX )

class Rand
{

public:

  static long lrand( long low, long high ) {
    return (long)(rand() * DIVRAND * (high - low)) + low;
  }
  
  static float frand( float low, float high ) {
    return ( (float) rand() * DIVRAND * ( high - low ) + low );
  }

  static Vector vrand( void ) {

    float x = Rand::frand( 0.0, 1.0 );
    float y = Rand::frand( 0.0, 1.0 );

    float xx, yy, offset, phi;

    x = 2*x - 1;
    y = 2*y - 1;
    
    if (y > -x) {               // Above y = -x
      if (y < x) {                // Below y = x
	xx = x;
	if (y > 0) {                // Above x-axis
	  /*
	   * Octant 1
	   */
	  offset = 0;
	  yy = y;
	} else {                    // Below and including x-axis
	  /*
	   * Octant 8
	   */
	  offset = (7*M_PI)/4;
	  yy = x + y;
	}
      } else {                    // Above and including y = x
	xx = y;
	if (x > 0) {                // Right of y-axis
	  /*
	   * Octant 2
	   */
	  offset = M_PI/4;
	  yy = (y - x);
	} else {                    // Left of and including y-axis
	  /*
	   * Octant 3
	   */
	  offset = (2*M_PI)/4;
	  yy = -x;
	}
      }
    } else {                    // Below and including y = -x
      if (y > x) {                // Above y = x
	xx = -x;
	if (y > 0) {                // Above x-axis
	  /*
	   * Octant 4
	   */
	  offset = (3*M_PI)/4;
	  yy = -x - y;
	} else {                    // Below and including x-axis
	  /*
	   * Octant 5
	   */
	  offset = (4*M_PI)/4;
	  yy = -y;
	}
      } else {                    // Below and including y = x
	xx = -y;
	if (x > 0) {                // Right of y-axis
	  /*
	   * Octant 7
	   */
	  offset = (6*M_PI)/4;
	  yy = x;
	} else {                    // Left of and including y-axis
	  if (y != 0) {
	    /*
	     * Octant 6
	     */
	    offset = (5*M_PI)/4;
	    yy = x - y;
	  } else {
	    /*
	     * Origin
	     */
	    return Vector( 0.0, 0.0, 1.0 );
	  }
	}
      }
    }
    
    float cost = 1 - xx*xx;
    assert(1 - cost*cost >= 0.0);
    float sint = sqrt(1 - cost*cost);
    
    phi = offset + (M_PI/4)*(yy/xx);

    return Vector( cos( phi )*sint, sin( phi )*sint, cost );

  }

};

#endif
#endif
