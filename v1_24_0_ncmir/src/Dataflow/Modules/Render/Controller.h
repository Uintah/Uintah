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
