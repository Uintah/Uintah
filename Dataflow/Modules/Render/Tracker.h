
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
