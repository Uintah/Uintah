/*------------------------------------------------------------------
 * receiver.h - Data structure for magnetic tracker receivers.
 *
 * Author: Milan Ikits (ikits@cs.utah.edu)
 * 
 * Copyright (C) 1999 
 * Center for Scientific Computing and Imaging
 * University of Utah
 *
 * Creation: 08/04/99
 * Last modification: 09/13/99
 * Comments:
 *------------------------------------------------------------------*/

#ifndef RECEIVER_H_
#define RECEIVER_H_

#include <GL/gl.h>

/* The orientation and position of the receiver are stored
   in a 4x4 homogeneous transformation matrix according to
   the OpenGL convention:

   [ R p ]   [ m[0] m[4] m[8]  m[12] ]
   [     ] = [ m[1] m[5] m[9]  m[13] ]
   [     ]   [ m[2] m[6] m[10] m[14] ]
   [ 0 1 ]   [ m[3] m[7] m[11] m[15] ]

   where R is a 3x3 rotation matrix in column vector format, 
   and p is a 3x1 position vector. 

   The orientation is also given by unit quaternion q. */

typedef struct ReceiverStruct {
  GLfloat m[16];
  GLfloat q[4];
} Receiver;

#endif /* RECEIVER_H_ */
