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
 *  FileReader.h
 *
 *  Written by:
 *   Keming Zhang 
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef SCIRun_Framework_FileReader_h
#define SCIRun_Framework_FileReader_h

#include <Core/CCA/spec/cca_sidl.h>

#define myUIPort FileReaderUIPort

//namespace SCIRun {
  

class FileReader;
class myUIPort : public virtual sci::cca::ports::UIPort {
public:
   virtual ~myUIPort(){}
   virtual int ui();
   void setParent(FileReader *com){this->com=com;}
   FileReader *com;
};

class myPDEDescriptionPort : public virtual sci::cca::ports::PDEDescriptionPort {
public:
   virtual ~myPDEDescriptionPort(){}
   virtual SSIDL::array1<double> getNodes();
   virtual SSIDL::array1<int> getBoundaries();
   virtual SSIDL::array1<int> getDirichletNodes();
   virtual SSIDL::array1<double> getDirichletValues();
   void setParent(FileReader *com){this->com=com;}
   FileReader *com;
};


class FileReader : public sci::cca::Component{
                
  public:
    FileReader();
    virtual ~FileReader();

    virtual void setServices(const sci::cca::Services::pointer& svc);
    
    SSIDL::array1<double> nodes;
    SSIDL::array1<int> boundaries;
    SSIDL::array1<int> dirichletNodes;
    SSIDL::array1<double> dirichletValues;
  private:

    FileReader(const FileReader&);
    FileReader& operator=(const FileReader&);
    myUIPort uiPort;
    myPDEDescriptionPort pdePort;
    sci::cca::Services::pointer services;
  };
//}




#endif
