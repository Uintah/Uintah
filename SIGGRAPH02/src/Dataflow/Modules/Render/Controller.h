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


#ifndef CONTROLLER_H_
#define CONTROLLER_H_

#define CONTROLLER_NONE    0
#define CONTROLLER_STYLUS  1
#define CONTROLLER_PINCH   2
#define CONTROLLER_I3STICK 3

#define NUM_CONTROLLER_RECEIVERS 2
#define CONTROLLER_ONE     0
#define CONTROLLER_LEFT    0
#define CONTROLLER_RIGHT   1

#include <Dataflow/Modules/Render/glMath.h>
#include <Dataflow/Modules/Render/SharedMemory.h>

namespace SCIRun {

class Controller {

public:

  int type;
  char arena[256];

  void *data;
  SharedMemory shmem;
  int receiver[NUM_CONTROLLER_RECEIVERS];
  GLfloat offset[NUM_CONTROLLER_RECEIVERS][16];

  Controller( void )
    {
      type = CONTROLLER_NONE;

      for( int i=0; i<NUM_CONTROLLER_RECEIVERS; i++ ) glEye(offset[i]);
    }
  ~Controller( void ) {}

};

} // End namespace SCIRun

#endif /* CONTROLLER_H_ */
