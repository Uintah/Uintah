
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

#include <PSECommon/Modules/Salmon/glMath.h>

namespace PSECommon {
namespace Modules {

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

}}

#endif /* CONTROLLER_H_ */
