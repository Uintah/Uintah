/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
