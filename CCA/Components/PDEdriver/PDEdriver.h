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
 *  PDEdriver.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Dec 2004
 *
 */

#ifndef CCA_Components_PDEdriver_h
#define CCA_Components_PDEdriver_h

#include <Core/CCA/spec/cca_sidl.h>

//#define myGoPort PDEdriverGoPort

namespace SCIRun {

class PDEdriver : public sci::cca::Component {
public:
  PDEdriver();
  virtual ~PDEdriver();
  virtual void setServices(const sci::cca::Services::pointer& svc);

private:
  PDEdriver(const PDEdriver&);
  PDEdriver& operator=(const PDEdriver&);

  sci::cca::Services::pointer services;
};

class myGoPort : public virtual sci::cca::ports::GoPort {
public:
  myGoPort(const sci::cca::Services::pointer &svc);
  virtual ~myGoPort(){}
  virtual int go();

private:
  void updateProgress(int counter);
  sci::cca::Services::pointer svc;
  sci::cca::ports::Progress::pointer pPtr;
};

class PDEComponentIcon : public virtual sci::cca::ports::ComponentIcon {
public:
  virtual ~PDEComponentIcon() {}
  virtual std::string getDisplayName();
  virtual std::string getDescription();
  virtual std::string getIconShape();
  virtual int getProgressBar();
};

}



#endif
