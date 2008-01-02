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
//  Trace.h: 
//
//    Turns on the ability to trace every malloc and free.  Output
//    defaults to stdout, but may be redirected to a file.
//
//    #included from Allocator.h (only if SCI_MALLOC_TRACE is defined).
//    However, it can be #include'd directly (to a file) if so desired.
//    Remember, only one of SCI_MALLOC_TRACE or !DISABLE_SCI_MALLOC
//    (ie, tracing or using SCI malloc) may be used at a time.
//
//    To receive full benefit of this code, you must use "scinew"
//    instead of "new" for all of your allocations.
//
//  Written by:
//   Author: J. Davison de St. Germain
//   C-SAFE
//   University of Utah
//   Date: Dec. 18, 2007
//
//  Copyright (C) 2007 - SCI Institute
//

//
// Please do NOT #include this file directly.  Just #include <Core/Malloc/Allocator.h>.
//

#ifndef MALLOC_TRACE_H
#define MALLOC_TRACE_H 1

#include <stdio.h>

/////////////////////////////////////////////////////////////////////////////////////
// Use 'scinew' instead of just 'new'.  This allows that tracing of
// the file name and line number of where the 'new' occurred.

#define scinew new( __FILE__, __LINE__ )
void* operator new(   size_t, const char * filename, int lineNumber );
void* operator new[]( size_t, const char * filename, int lineNumber );


namespace SCIRun {

/////////////////////////////////////////////////////////////////////////////////////
//
// MallocTraceInfo_S is used to log memory events to a specified file.
//

struct MallocTraceInfo_S
{
public:
  MallocTraceInfo_S();
  ~MallocTraceInfo_S();

  // Tracing defaults to being on.  Use this procedure to turn it off if so desired.
  void setTracingState( bool on );

  // Sets the name of (and opens) the output file that will be used to
  // log memory events.  If a file is already open, then it will be closed.
  // If a file of the same name already exists, it will be overwritten.
  // If an output filename is not specified, the output will go to the console.
  // The 'info' string will be copied to the top of the output file.
  void setOutputFilename( const char * outputFilename, const char * info );

  // Set's the filename and line number for output purposes.
  // Always returns 'false' for use in wrapper macro.
  bool   set( const char * file, int line ) { file_ = file; lineNumber_ = line; return false; }

  // Sets a (general) tag that will be displayed with each malloc
  // print.  This is useful in helping trace mallocs from external
  // libraries that do not use "scinew".  Returns the previous tag.
  const char * set( const char * tag ) {
    const char * oldTag = tag_;
    tag_ = tag; 
    return oldTag;
  }

  // Allows the clearing of the trace information.  ResetAll() might
  // be useful to the user of this facility, but probably isn't.  (You
  // probably should just set() a new general tag.)
  // ResetFileAndLine() is used internally after each print out and
  // thus most likely is of no use to the user.  (It is 'public' so
  // that new() can use it.)
  void   resetAll() { file_ = NULL; lineNumber_ = -1; tag_ = NULL; }
  void   resetFileAndLine() { file_ = NULL; lineNumber_ = -1; }

  // Prints out the current trace information (such as general tag,
  // filename, line number, memory address, etc).   If size == -1, then don't print a size value.
  void   print( const char * infoString, void * addr, int size = -1 );

  // Prints out the status of the trace variable... Used to avoid
  // recompiling (or more accurately, re-linking) the entire world
  // when debugging the trace facility.
  void   status();

private:

  bool         doTrace_;

  int          mallocCount_; // Amount of memory malloc'd before file output turned on.

  const char * outputFilename_;
  const char * tag_;
  
  FILE * fp_;

  const char * file_;       // Name of file that new/malloc was called from.
  int          lineNumber_; // Line number of new/malloc (or generic position information)

};

/////////////////////////////////////////////////////////////////////////////////////

// Use this variable to access trace functions:

extern MallocTraceInfo_S mallocTraceInfo;

// This function allows us to take advantage of the fact that some
// Allocator code tagging already is present in some of SCIRun.  Note,
// it is not using the Allocator.h function... we are just overloading
// it for our use (as Allocator will not be on if Trace is on).

const char * AllocatorSetDefaultTagMalloc( const char * tag );

// Doesn't do anything right now...
void         AllocatorMallocStatsAppendNumber( int worldRank );

/////////////////////////////////////////////////////////////////////////////////////

} // end namespace SCIRun

#endif // MALLOC_TRACE_H
