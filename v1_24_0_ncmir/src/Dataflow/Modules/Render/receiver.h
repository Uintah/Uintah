/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
