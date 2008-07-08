/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2007 Scientific Computing and Imaging Institute,
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

//
//  Trace.cc: 
//
//  Written by:
//   Author: J. Davison de St. Germain
//   C-SAFE
//   University of Utah
//   Date: Dec. 18, 2007
//
//  Copyright (C) 2007 - SCI Institute
//


#include <Core/Malloc/Trace.h>

#include <malloc.h>
#include <cstdlib>
#include <cerrno>

using namespace std;
using namespace SCIRun;

/////////////////////////////////////////////////////////////////////////////////////
// mallocTraceInfo is used (globally) to access trace functionality.

MallocTraceInfo_S SCIRun::mallocTraceInfo;

/////////////////////////////////////////////////////////////////////////////////////

MallocTraceInfo_S::MallocTraceInfo_S() :
  doTrace_( true ), mallocCount_( 0 ), 
  outputFilename_( "" ), tag_( NULL ), fp_( NULL ), file_( NULL ), lineNumber_( -1 )
{
}

MallocTraceInfo_S::~MallocTraceInfo_S()
{
  if( fp_ != NULL ) {
    // Must set fp_ to NULL before the fclose, otherwise during the
    // fclose, some free's happen and the system tries to print them
    // to fp_, which isn't valid at that point (which causes a
    // segfault).
    FILE * tmpFp = fp_;
    fp_ = NULL;
    fclose( tmpFp );
  }
}

void
MallocTraceInfo_S::setOutputFilename( const char * outputFilename, const char * info )
{
  if( fp_ != NULL ) {
    fclose( fp_ );
  }
  outputFilename_ = outputFilename;

  fp_ = fopen( outputFilename_, "w" );

  // Store mallocCount_ in tmp var after fopen in order to account for memory allocated by fopen.
  int tmpMallocCount = mallocCount_;
  mallocCount_ = 0;

  if( fp_ == NULL ) {
    printf( "ERROR (%d) in opening file '%s'\n", errno, outputFilename_ );
    mallocCount_ = tmpMallocCount;
  }
  else {
    fprintf( fp_, "Fix me... save the date/time here.\n" );
    fprintf( fp_, "Memory malloc'd before file logging turned on: %d.\n\n", tmpMallocCount );
    if( info != NULL ) {
      fprintf( fp_, "%s\n", info );
    }
  }
}

void
MallocTraceInfo_S::setTracingState( bool on )
{
  doTrace_ = on;
}

void
MallocTraceInfo_S::print( const char * infoString, void * addr, int size /* = -1 */ )
{ 
  if( !doTrace_ ) return;

  FILE * tmpFp = stdout;
  if( fp_ != NULL ) {
    tmpFp = fp_;
  } else {
    if( size > 0 ) {
      mallocCount_ += size;
    }
  }

  if( size == -1 ) {
    fprintf( tmpFp, "%s - %p", infoString, addr );
  } else {
    fprintf( tmpFp, "%s - %p - %u bytes", infoString, addr, size );
  }
  
  if( file_ != NULL ) {
    fprintf( tmpFp, " -- %s (%d)", file_, lineNumber_ );
  }
  if( tag_ != NULL ) {
    fprintf( tmpFp, " -- %s", tag_ );
  }
  fprintf( tmpFp, "\n" );
}

//////////////////////////////////////////////////////////////////////////////////////////

const char *
SCIRun::AllocatorSetDefaultTagMalloc( const char * tag )
{
  return mallocTraceInfo.set( tag );
}

// Not sure what this is supposed to do...
void
SCIRun::AllocatorMallocStatsAppendNumber( int worldRank )
{
}

//////////////////////////////////////////////////////////////////////////////////////////

static void *(*original_malloc_hook) ( size_t, const void* );
static void (*original_free_hook) ( void * ptr, const void* );


static void *
my_malloc_hook( size_t size,const void *caller )
{
  //printf("mh: %p, my_mh: %p\n", __malloc_hook, my_malloc_hook );

  void *result;
  __malloc_hook = original_malloc_hook;
  result = malloc( size );
  __malloc_hook = my_malloc_hook;
 
  mallocTraceInfo.print( "malloc", result, size );

  //printf("end my_malloc hook,  MH: %p, my_mh: %p\n", __malloc_hook, my_malloc_hook );

  return result;
}

static void
my_free_hook( void * ptr,const void *caller )
{
  if( __malloc_hook == NULL ) {
    printf("WARNING!!!! __malloc_hook is NULL... Resetting!!!\n");
    __malloc_hook = my_malloc_hook;
  }

  //printf("Begin my_free_hook: MH: %p, my_mh: %p\n", __malloc_hook, my_malloc_hook );
  __free_hook = original_free_hook;
  free( ptr );
  __free_hook = my_free_hook;
  mallocTraceInfo.print( "free  ", ptr );
  //printf("End my_free_hook: MH: %p, my_mh: %p\n", __malloc_hook, my_malloc_hook );
}

//////////////////////////////////////////////////////////////////////////////////////////////

void*
operator new( size_t size, const char * filename, int lineNumber )
{
  //printf("new - ");
  //printf("mh: %p, my_mh: %p\n", __malloc_hook, my_malloc_hook );
  mallocTraceInfo.set( filename, lineNumber );
  void * ptr = malloc( size );
  mallocTraceInfo.resetFileAndLine();
  return ptr;
}

void*
operator new[]( size_t size, const char * filename, int lineNumber )
{
  //printf("new[]\n");
  mallocTraceInfo.set( filename, lineNumber );
  void * ptr = malloc( size );
  mallocTraceInfo.resetFileAndLine();
  return ptr;
}

//////////////////////////////////////////////////////////////////////////////////////////

static
void
my_init_hook()
{
  original_malloc_hook = __malloc_hook;
  original_free_hook   = __free_hook;

  __malloc_hook = my_malloc_hook;
  __free_hook = my_free_hook;
}

/* Override initializing hook from the C library. */
void (*__malloc_initialize_hook)(void) = my_init_hook;

//////////////////////////////////////////////////////////////////////////////////////////
