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


#ifndef TRACKER_H_
#define TRACKER_H_

#define TRACKER_NONE    0
#define TRACKER_FASTRAK 1
#define TRACKER_FOB     2

#include <Dataflow/Modules/Render/SharedMemory.h>

namespace SCIRun {

class Tracker {

public:

  int type;
  char arena[256];
  void* data;
  SharedMemory shmem;

  Tracker( void ) { type = TRACKER_NONE; }
  ~Tracker( void ) {}
};

} // End namespace SCIRun

#endif /* TRACKER_H_ */
