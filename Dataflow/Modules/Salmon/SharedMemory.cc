
#include <unistd.h>
#include <malloc.h>
#include <fcntl.h>

#include <sys/types.h>
#include <sys/prctl.h>
#include <sys/time.h>
#include <sys/stat.h>

#include <string.h>

#include <iostream>
using std::cerr;
using std::endl;

#include "SharedMemory.h"

namespace PSECommon {
namespace Modules {

/* 
 * Create shared memory region based on arena file given.
 */

int SharedMemory :: init( char* file, int numproc, int address, int arenasize,
		   void** data, int size )
{
  int h;

  if( file == 0 )
    {
      cerr << "Error: No arena file name given\n";
      return(-1);
    }

  strcpy(arenafile, file);

/* Check if arena file exists. */

  if( (h = open(file, O_RDONLY)) > 0 )
    {
      close(h);
      cerr << "Error: Arena file " << file << " already exists.\n";
      return(-1);
    }

/* Create arena. */

  usconfig(CONF_INITUSERS, numproc);
  usconfig(CONF_ATTACHADDR, address);
  usconfig(CONF_INITSIZE, arenasize);

  if( (arena = usinit(file)) == 0 )
    {
      cerr << "Error: Cannot initialize arena.\n";
      return(-1);
    }

/* Allocate shared space for the administration record. */

  if( (shared = (SharedData*)usmalloc(sizeof(SharedData), arena)) < 0 )
    {
      cerr << "Error: Cannot allocate shared memory administration space.\n";
      return(-1);
    }

/* Allocate shared space for common data. */

  if( (shared->data = usmalloc(size, arena)) < 0 )
    {
      cerr << "Error: Cannot allocate shared memory data space.\n";
      return(-1);
    }

  *data = shared->data;

/* Create semaphore. */

  shared->sema = usnewsema(arena, 1);

/* Put shared data address into arena. */

  usputinfo(arena, shared);

  return(0);
}

/* 
 * Destroy shared memory region created by init. Remove the arena file, too.
 */

void SharedMemory :: destroy( void )
{
  usfree(shared->data, arena);
  usfree(shared, arena);
  unlink(arenafile);
}

/* 
 * Attach shared memory region with arena file given. 
 */

int SharedMemory :: attach( char* file, void** data )
{
  if( file == 0 )
    {
      cerr << "Error: No arena file name given\n";
      return(-1);
    }

/* Check if arena file exists. */ 

  if( open(file, O_RDONLY) < 0 )
    {
      cerr << "Error: Arena file " << file << " does not exist.\n";
      return(-1);
    }

/* Attach arena. */

  if( (arena = usinit(file)) == NULL )
    {
      cerr << "Error: Cannot attach to arena.\n";
      return(-1);
    }

/* Get shared data address. */

  shared = (SharedData*)usgetinfo(arena);

  *data = shared->data;
  
  return(0);
}

/*
 * Detach shared memory area.
 */

void SharedMemory :: detach( void )
{
  usdetach(arena);
}

/*
 * Lock data in shared memory data structure.
 */

void SharedMemory :: lock( void )
{
  uspsema(shared->sema);
}

/*
 * Unlock data in shared memory data structure.
 */

void SharedMemory :: unlock( void )
{
  usvsema(shared->sema);
}

}}
