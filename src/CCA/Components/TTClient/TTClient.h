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
 *  TTClient.h
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#ifndef SCIRun_Framework_TTClient_h
#define SCIRun_Framework_TTClient_h

#include <CCA/Components/TableTennis/TableTennis_sidl.h>
#include <Core/CCA/spec/cca_sidl.h>

class TTClient;

class ttUIPort : public sci::cca::ports::UIPort {
public:
  virtual ~ttUIPort(){}
  virtual int ui();
  void setParent(TTClient *TTCl) { this->TTCl = TTCl; }
  TTClient *TTCl;
};

class ttGoPort : public sci::cca::ports::GoPort {
public:
  virtual ~ttGoPort(){}
  virtual int go();
  void setParent(TTClient *TTCl) { this->TTCl = TTCl; }
  TTClient *TTCl;
};


class TTClient : public sci::cca::Component {

public:
  TTClient();
  virtual ~TTClient();
  virtual void setServices(const sci::cca::Services::pointer& svc);
  sci::cca::Services::pointer& getServices() { return services; }
  inline int getCount() { return count; }
  inline void setCount(int i) { count = i; }

private:
  TTClient(const TTClient&);
  TTClient& operator=(const TTClient&);
  ttUIPort::pointer uiPortPtr;
  ttGoPort::pointer goPortPtr;
  sci::cca::Services::pointer services;
  int count;
};


#endif
