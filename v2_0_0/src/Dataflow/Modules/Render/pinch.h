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
 * pinch.h - Data structure for the Fakespace Pinch Gloves.
 *
 * Author: Milan Ikits (ikits@cs.utah.edu)
 * 
 * Copyright (C) 1999 
 * Center for Scientific Computing and Imaging
 * University of Utah
 *
 * Creation: 06/30/99
 * Last modification: 06/30/99
 * Comments:
 *------------------------------------------------------------------*/

#ifndef PINCH_H_
#define PINCH_H_

#define PINCH_NONE 0x00

#define PINCH_LEFT_THUMB_LEFT_INDEX  0x01
#define PINCH_LEFT_THUMB_LEFT_MIDDLE 0x02
#define PINCH_LEFT_THUMB_LEFT_RING   0x03
#define PINCH_LEFT_THUMB_LEFT_PINKY  0x04

#define PINCH_LEFT_MASK 0x07

#define PINCH_RIGHT_THUMB_RIGHT_INDEX  0x08
#define PINCH_RIGHT_THUMB_RIGHT_MIDDLE 0x10
#define PINCH_RIGHT_THUMB_RIGHT_RING   0x18
#define PINCH_RIGHT_THUMB_RIGHT_PINKY  0x20

#define PINCH_RIGHT_MASK 0x38

#define PINCH_LEFT_THUMB_RIGHT_THUMB  0x40
#define PINCH_LEFT_THUMB_RIGHT_INDEX  0x80
#define PINCH_LEFT_THUMB_RIGHT_MIDDLE 0x100
#define PINCH_LEFT_THUMB_RIGHT_RING   0x200
#define PINCH_LEFT_THUMB_RIGHT_PINKY  0x400

#define PINCH_LEFT_INDEX_RIGHT_THUMB  0x800
#define PINCH_LEFT_INDEX_RIGHT_INDEX  0x1000
#define PINCH_LEFT_INDEX_RIGHT_MIDDLE 0x2000
#define PINCH_LEFT_INDEX_RIGHT_RING   0x4000
#define PINCH_LEFT_INDEX_RIGHT_PINKY  0x8000

#define PINCH_LEFT_MIDDLE_RIGHT_THUMB  0x10000
#define PINCH_LEFT_MIDDLE_RIGHT_INDEX  0x20000
#define PINCH_LEFT_MIDDLE_RIGHT_MIDDLE 0x40000
#define PINCH_LEFT_MIDDLE_RIGHT_RING   0x80000
#define PINCH_LEFT_MIDDLE_RIGHT_PINKY  0x100000

#define PINCH_LEFT_RING_RIGHT_THUMB  0x200000
#define PINCH_LEFT_RING_RIGHT_INDEX  0x400000
#define PINCH_LEFT_RING_RIGHT_MIDDLE 0x800000
#define PINCH_LEFT_RING_RIGHT_RING   0x1000000
#define PINCH_LEFT_RING_RIGHT_PINKY  0x2000000

#define PINCH_LEFT_PINKY_RIGHT_THUMB  0x4000000
#define PINCH_LEFT_PINKY_RIGHT_INDEX  0x8000000
#define PINCH_LEFT_PINKY_RIGHT_MIDDLE 0x10000000
#define PINCH_LEFT_PINKY_RIGHT_RING   0x20000000
#define PINCH_LEFT_PINKY_RIGHT_PINKY  0x40000000

typedef unsigned int Gesture;

typedef struct PinchStruct {
  Gesture gesture;
} Pinch;

#endif /* PINCH_H_ */
