
#ifndef CONTROLLER_H_
#define CONTROLLER_H_

#define CONTROLLER_NONE    0
#define CONTROLLER_STYLUS  1
#define CONTROLLER_PINCH   2
#define CONTROLLER_I3STICK 3

#define NUM_CONTROLLER_RECEIVERS 2
#define CONTROLLER_LEFT    0
#define CONTROLLER_RIGHT   1

namespace PSECommon {
namespace Modules {

class Controller {

public:

  int type;
  char arena[256];

  void *data, *prev;
  SharedMemory shmem;
  int receiver[NUM_CONTROLLER_RECEIVERS];
  GLfloat mprev[NUM_CONTROLLER_RECEIVERS][16];

  Controller( void ) { type = CONTROLLER_NONE; }
  ~Controller( void ) {}

};

}}

#endif /* CONTROLLER_H_ */
