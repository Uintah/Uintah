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


/*
 *  AllocOS.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Malloc_AllocOS_h
#define Malloc_AllocOS_h 1

#include <stdlib.h>

namespace SCIRun {

struct OSHunk {
    static OSHunk* alloc(size_t size, bool returnable);
    static void free(OSHunk*);
    void* data;
    OSHunk* next;

    int ninuse;
    size_t spaceleft;
    void* curr;
    size_t len;
    bool returnable;
    double align;
};

} // End namespace SCIRun

#endif
