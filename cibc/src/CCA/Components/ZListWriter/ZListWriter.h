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

#ifndef SCIRun_CCA_Component_ZListWriter_h
#define SCIRun_CCA_Component_ZListWriter_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>

class ZListWriter;

class ZLUIPort : public virtual sci::cca::ports::UIPort {
public:
    ZLUIPort(const sci::cca::Services::pointer& svc) { services = svc; }
    virtual ~ZLUIPort(){}
    virtual int ui();
    void setParent(ZListWriter *com) { this->com = com; }
private:
    ZListWriter *com;       
    sci::cca::Services::pointer services;
};

// class ZListPort : public virtual sci::cca::ports::ZListPort {
// public:
//     virtual ~ZListPort(){}
//     virtual SSIDL::array1<double> getList();
//     void setParent(ZListWriter *com){ this->com = com; }
// private:
//     ZListWriter *com;       
// };

class ZLComponentIcon : public virtual sci::cca::ports::ComponentIcon {
public:
  virtual ~ZLComponentIcon() {}

  virtual std::string getDisplayName();
  virtual std::string getDescription();
  virtual std::string getIconShape();
  virtual int getProgressBar();
private:
  static const int STEPS = 10;
};


class ZListWriter : public sci::cca::Component {
public:
    std::vector<double> datalist;   
    ZListWriter();
    virtual ~ZListWriter();
    virtual void setServices(const sci::cca::Services::pointer& svc);
private:
    ZListWriter(const ZListWriter&);
    ZListWriter& operator=(const ZListWriter&);
    sci::cca::Services::pointer services;
};

#endif
