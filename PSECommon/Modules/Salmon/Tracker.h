
#ifndef TRACKER_H_
#define TRACKER_H_

#define TRACKER_NONE    0
#define TRACKER_FASTRAK 1
#define TRACKER_FOB     2

#include <PSECommon/Modules/Salmon/SharedMemory.h>

namespace PSECommon {
namespace Modules {

class Tracker {

public:

  int type;
  char arena[256];
  void* data;
  SharedMemory shmem;

  Tracker( void ) { type = TRACKER_NONE; }
  ~Tracker( void ) {}
};

}}

#endif /* TRACKER_H_ */
