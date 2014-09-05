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
