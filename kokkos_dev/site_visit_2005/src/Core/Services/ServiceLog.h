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


// ServiceLog.h


#ifndef CORE_SERVICES_SERVICELOG_H
#define CORE_SERVICES_SERVICELOG_H 1

#include <Core/Services/ServiceBase.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Containers/LockingHandle.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
	
class ServiceLog;
typedef LockingHandle<ServiceLog> ServiceLogHandle;
	
	
class ServiceLog : public ServiceBase {

public:
  // Create a log file, that can used by multiple threads without
  // the log becoming a mesh
  ServiceLog(std::string filename);

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
