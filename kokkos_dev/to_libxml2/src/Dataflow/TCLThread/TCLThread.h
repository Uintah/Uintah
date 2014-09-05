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

#ifndef SCI_project_TCLThread_h
#define SCI_project_TCLThread_h 1

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>

namespace SCIRun {

class Network;
class TCLInterface;

class TCLThread : public Runnable
{
public:
  TCLThread(int argc, char* argv[], Network* net, int startnetno);
  inline TCLInterface* getTclInterface() { return gui;};
  virtual void run();
  void mainloop_waitstart();
  void release_mainloop();
  void mainloop_wait();
  void startNetworkEditor();
private:
  bool check_for_newer_scirunrc();
  int argc;
  char** argv;
  Network* net;
  int startnetno;
  Semaphore cont;
  Semaphore start;
  TCLInterface* gui;
};

}

#endif
