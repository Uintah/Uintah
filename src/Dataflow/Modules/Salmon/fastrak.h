/*------------------------------------------------------------------
 * fastrak.h - Data structure for the Polhemus Fastrak.
 *
 * Author: Milan Ikits (ikits@cs.utah.edu)
 * 
 * Copyright (C) 1999 
 * Center for Scientific Computing and Imaging
 * University of Utah
 *
 * Creation: 06/13/99
 * Last modification: 08/04/99
 * Comments:
 *------------------------------------------------------------------*/

#ifndef FASTRAK_H_
#define FASTRAK_H_

#include "receiver.h"

#define FASTRAK_MAX_RECEIVER 4

typedef int Stylus;

#define STYLUS_OFF 0
#define STYLUS_ON  1

typedef struct FastrakStruct {
  Receiver receiver[FASTRAK_MAX_RECEIVER];
  Stylus stylus;     /* stylus state 0: off, 1: on */
} Fastrak;

#endif /* FASTRAK_H_ */
