
#ifndef SHAREDMEMORY_H_
#define SHAREDMEMORY_H_

#include <ulocks.h>

namespace PSECommon {
namespace Modules {

typedef struct SharedDataStruct {
  usema_t *sema;
  void* data;
} SharedData;

class SharedMemory {
protected:
  char arenafile[256];
  usptr_t *arena;
  SharedData *shared;

public:
  SharedMemory( void ) {}
  ~SharedMemory( void ) {}

  int init( char*, int, int, int, void**, int );
  void destroy( void );

  int attach( char*, void** );
  void detach( void );
  
  void lock( void );
  void unlock( void );
};

}}

#endif /* SHAREDMEMORY_H_ */
