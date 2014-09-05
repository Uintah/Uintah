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
 *  Hello.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#ifndef SCIRun_CCA_Components_Hello_h
#define SCIRun_CCA_Components_Hello_h

#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/Hello/Hello_sidl.h>

namespace SCIRun {

class Hello;

class HelloUIPort : public sci::cca::ports::UIPort {
public:
  virtual ~HelloUIPort() {}
  int ui();
  void setParent(Hello *com) { this->com = com; }
  Hello *com;
};

class HelloGoPort : public sci::cca::ports::GoPort {
public:
  virtual ~HelloGoPort() {}

  HelloGoPort(const sci::cca::Services::pointer& svc) : services(svc) {}
  int go();
  void setParent(Hello *com) { this->com = com; }
  Hello *com;

private:
  sci::cca::Services::pointer services;
};

class HelloComponentIcon : public sci::cca::ports::ComponentIcon {
public:
  HelloComponentIcon() {}
  virtual ~HelloComponentIcon() {}

  virtual std::string getDisplayName();
  virtual std::string getDescription();
  virtual std::string getIconShape();
  virtual int getProgressBar();
  static const int STEPS = 50;
};

// demonstrate use of Go port, UI port and ComponentRelease
class Hello : public sci::cca::Hello {
public:
  Hello();
  virtual ~Hello();
  virtual void setServices(const sci::cca::Services::pointer& svc);
  virtual void releaseServices(const sci::cca::Services::pointer& svc);
  std::string text;

private:
  void setCommunicator(int comm) {
    //MPI_COMM_COM=MPI_COMM_WORLD; //*(MPI_Comm*)(comm);
  }
  Hello(const Hello&);
  Hello& operator=(const Hello&);

  sci::cca::Services::pointer services;
};
  
} //namespace SCIRun


#endif
