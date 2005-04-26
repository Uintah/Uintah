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
 *  CorbaUIPort.cc: CCA-style Interface to old TCL interfaces
 *
  *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/Corba/CorbaUIPort.h>
#include <SCIRun/Corba/CorbaComponentInstance.h>

#include <iostream>

namespace SCIRun {

/**
 * \class CorbaUIThread
 *
 * ?
 */
class CorbaUIThread : public Runnable {
public:
  CorbaUIThread(CorbaComponentInstance* ci);
  ~CorbaUIThread() {}
  /** ? */
  void run();
private:
  CorbaComponentInstance* ci;
};

CorbaUIThread::CorbaUIThread(CorbaComponentInstance* ci)
  :ci(ci)
{
}

void CorbaUIThread::run()
{
  ci->getComponent()->popupUI();
}

CorbaUIPort::CorbaUIPort(CorbaComponentInstance* ci)
  : ci(ci)
{
}

CorbaUIPort::~CorbaUIPort()
{
}

int 
CorbaUIPort::ui()
{
  if(ci->getComponent()->isThreadedUI()){
  Thread* t = new Thread(new CorbaUIThread(ci), "Corba UI Thread", 0);
  t->detach();
  }else{
  return ci->getComponent()->popupUI();
  }
  //return 0 success, -1 fatal error, 
  //other values for other errors.
  return 0;
}

} // end namespace SCIRun
