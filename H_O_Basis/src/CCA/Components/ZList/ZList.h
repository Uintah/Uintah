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
 *  ZList.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   March 2002
 *
 */

#ifndef SCIRun_Framework_ZList_h
#define SCIRun_Framework_ZList_h

#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/PIDL.h>
#include "ListForm.h"



//namespace SCIRun {


//#define ImUIPort1 
//ImUIPort

class ZList;

  class ImUIPort1 : public virtual sci::cca::ports::UIPort {

  public:
    virtual ~ImUIPort1(){}
    virtual int ui();
    void setParent(ZList *com){this->com=com;}
  private:
    ZList *com;	   	
  };

  class ImZListPort : public virtual sci::cca::ports::ZListPort {

  public:
    ImZListPort(){};	
    virtual ~ImZListPort(){};
    virtual SSIDL::array1<double> getList();
    void setParent(ZList *com){this->com=com;}
  private:
    ZList *com;	   	
  };


class ZList : public sci::cca::Component{

  public:
    std::vector<double> datalist; 	
    ZList();
    virtual ~ZList();

    virtual void setServices(const sci::cca::Services::pointer& svc);
  private:

    ZList(const ZList&);
    ZList& operator=(const ZList&);
    ImUIPort1 uiport;
    ImZListPort listport;
    sci::cca::Services::pointer services;
  };
//}




#endif
