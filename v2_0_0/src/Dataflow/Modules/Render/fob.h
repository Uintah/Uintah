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
