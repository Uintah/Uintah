/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
class myUIPort : public virtual gov::cca::ports::UIPort {
public:
   virtual ~myUIPort(){}
   virtual int ui();
   void setParent(FileReader *com){this->com=com;}
   FileReader *com;
};

class myPDEDescriptionPort : public virtual gov::cca::ports::PDEDescriptionPort {
public:
   virtual ~myPDEDescriptionPort(){}
   virtual SIDL::array1<double> getNodes();
   virtual SIDL::array1<int> getBoundaries();
   virtual SIDL::array1<int> getDirichletNodes();
   virtual SIDL::array1<double> getDirichletValues();
   void setParent(FileReader *com){this->com=com;}
   FileReader *com;
};


class FileReader : public gov::cca::Component{
                
  public:
    FileReader();
    virtual ~FileReader();

    virtual void setServices(const gov::cca::Services::pointer& svc);
    
    SIDL::array1<double> nodes;
    SIDL::array1<int> boundaries;
    SIDL::array1<int> dirichletNodes;
    SIDL::array1<double> dirichletValues;
  private:

    FileReader(const FileReader&);
    FileReader& operator=(const FileReader&);
    myUIPort uiPort;
    myPDEDescriptionPort pdePort;
    gov::cca::Services::pointer services;
  };
//}




#endif
