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



/*
 *  ProgressReporter.h:
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 */


#ifndef SCIRun_Core_Util_ProgressReporter_h
#define SCIRun_Core_Util_ProgressReporter_h

#include <Core/share/share.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Timer.h>
#include <Core/Thread/Mutex.h>

namespace SCIRun {

  //using namespace std;

class SCICORESHARE ProgressReporter 
{
public:
  typedef enum {Starting, Compiling, CompilationDone, Done } ProgressState;
  ProgressReporter() :
    current_(0),
    progress_lock_("ProgressReporter")
  {}
  virtual ~ProgressReporter();

  virtual void error(const std::string& msg)      { std::cerr << "Error: " << msg << std::endl; }
  virtual void warning(const std::string& msg)    { std::cerr << "Warning: " << msg << std::endl; }
  virtual void remark(const std::string& msg)     { std::cerr << "Remark: " << msg << std::endl; }
  virtual void postMessage(const std::string&msg) { std::cerr << "Msg: " << msg << std::endl; }

  virtual std::ostream &msgStream() { return std::cerr; }
  virtual void msgStream_flush() {}


  virtual void report_progress( ProgressState ) {}

  virtual void update_progress(double) {}
  virtual void update_progress(double, Timer &) {}
  virtual void update_progress(int, int) {}
  virtual void update_progress(int, int, Timer &) {}
  virtual void accumulate_progress(int) {}
protected:
  // accumulation storage;
  int                current_;
  int                max_;
  Mutex              progress_lock_;
};


} // Namespace SCIRun

#endif
