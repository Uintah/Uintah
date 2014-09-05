/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// LogFile.h


#ifndef CORE_UTIL_LOGFILE_H
#define CORE_UTIL_LOGFILE_H 1

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>

#include <fstream>
#include <string>

#include <Core/Util/share.h>
namespace SCIRun {
	
class LogFile;
typedef LockingHandle<LogFile> LogFileHandle;
	
	
class SCISHARE LogFile {

public:
  // Create a log file, that can used by multiple threads without
  // the log becoming a mesh
  LogFile(std::string filename);

  // Since we used a ofstream, this one is automatically closed when the
  // object is destroyed.
	
  // Write a message into the log
  void putmsg(std::string msg);

public:
  Mutex lock;
  int   ref_cnt;
  
private:
  std::ofstream logfile_;
  bool haslog_;
};

}

#endif
