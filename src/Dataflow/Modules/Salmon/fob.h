/*------------------------------------------------------------------
 * fob.h - Data structure for the Ascension Flock of Birds.
 *
 * Author: Milan Ikits (ikits@cs.utah.edu)
 * 
 * Copyright (C) 1999 
 * Center for Scientific Computing and Imaging
 * University of Utah
 *
 * Creation: 08/02/99
 * Last modification: 08/04/99
 * Comments:
 *------------------------------------------------------------------*/

#ifndef FOB_H_
#define FOB_H_

#include "receiver.h"

#define FOB_MAX_RECEIVER 14

typedef struct BirdStruct {
  Receiver receiver[FOB_MAX_RECEIVER];
} FoB;

#endif /* FOB_H_ */
