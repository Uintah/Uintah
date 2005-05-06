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



#include <unistd.h>

#ifndef __APPLE__
#  include <malloc.h>
#endif

#include <fcntl.h>
#include <sys/types.h>

#ifdef __sgi
#  include <sys/prctl.h>
#endif

#include <sys/time.h>
#include <sys/stat.h>

#include <string.h>

#include <iostream>
using std::cerr;
using std::endl;

#include <Dataflow/Modules/Render/SharedMemory.h>

namespace SCIRun {

/* 
 * Create shared memory region based on arena file given.
 */

int SharedMemory :: init( char* file, int numproc, int address, int arenasize,
		   void** data, int size )
{
#ifdef __sgi
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
#endif

  return(0);
}

/* 
 * Destroy shared memory region created by init. Remove the arena file, too.
 */

void SharedMemory :: destroy( void )
{
#ifdef __sgi
  usfree(shared->data, arena);
  usfree(shared, arena);
  unlink(arenafile);
#endif
}

/* 
 * Attach shared memory region with arena file given. 
 */

int SharedMemory :: attach( char* file, void** data )
{
#ifdef __sgi
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
#endif

  return(0);
}

/*
 * Detach shared memory area.
 */

void SharedMemory :: detach( void )
{
#ifdef __sgi
  usdetach(arena);
#endif
}

/*
 * Lock data in shared memory data structure.
 */

void SharedMemory :: lock( void )
{
#ifdef __sgi
  uspsema(shared->sema);
#endif
}

/*
 * Unlock data in shared memory data structure.
 */

void SharedMemory :: unlock( void )
{
#ifdef __sgi
  usvsema(shared->sema);
#endif
}

} // End namespace SCIRun
