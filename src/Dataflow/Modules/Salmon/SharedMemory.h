
#ifndef SHAREDMEMORY_H_
#define SHAREDMEMORY_H_

#ifdef __sgi
#include <ulocks.h>
#endif

namespace PSECommon {
namespace Modules {

typedef struct SharedDataStruct {
#ifdef __sgi
  usema_t *sema;
#endif
  void* data;
} SharedData;

class SharedMemory {
protected:
  char arenafile[256];
#ifdef __sgi
  usptr_t *arena;
#endif
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
