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
 *  Hello.cc:
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#include <CCA/Components/FileReader/FileReader.h>
#include <iostream>
#include <fstream>
#include <CCA/Components/Builder/QtUtils.h>

#include <qfiledialog.h>
#include <qmessagebox.h>



using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_FileReader()
{
  return sci::cca::Component::pointer(new FileReader());
}


FileReader::FileReader()
{
  uiPort.setParent(this);
  pdePort.setParent(this);
}

FileReader::~FileReader()
{
  cerr << "called ~FileReader()\n";
}

void FileReader::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...  

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  myUIPort::pointer uip(&uiPort);
  myPDEDescriptionPort::pointer pdep(&pdePort);
  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  svc->addProvidesPort(pdep,"pde","sci.cca.ports.PDEDescriptionPort", props);
  // Remember that if the PortInfo is created but not used in a call to the svc object
  // then it must be freed.
  // Actually - the ref counting will take care of that automatically - Steve
}

int myUIPort::ui() 
{
  QString fn = QFileDialog::getOpenFileName(
	    "./","PDE Description Files(*.pde)");
  if(fn.isNull()) return 1;
  ifstream is(fn);
  com->nodes.clear();
  com->boundaries.clear();
  com->dirichletNodes.clear();
  com->dirichletValues.clear();
  while(true){
    std::string name;
    is>>name;
    if(name=="node"){
      int cnt;
      is>>cnt;
      for(int i=0; i<cnt; i++){
	double x, y;
	is>>x>>y;
	com->nodes.push_back(x);
	com->nodes.push_back(y);
      }
    }
    else if(name=="boundary"){
      int cnt;
      is>>cnt;
      for(int i=0; i<cnt; i++){
	int index;
	is>>index;
	com->boundaries.push_back(index);
      }
    }
    else if(name=="dirichlet"){
      int cnt;
      is>>cnt;
      for(int i=0; i<cnt; i++){
	int index;
	is>>index;
	com->dirichletNodes.push_back(index);
      }
      for(int i=0; i<cnt; i++){
	double value;
	is>>value;
	com->dirichletValues.push_back(value);
      }
    }
    else if(name=="end") break;  
  }

  cerr<<com->nodes.size()<<endl;
  cerr<<com->boundaries.size()<<endl;
  cerr<<com->dirichletNodes.size()<<endl;
  cerr<<com->dirichletValues.size()<<endl;

  return 0;
}


SSIDL::array1<double> myPDEDescriptionPort::getNodes() 
{
  return com->nodes;
}

SSIDL::array1<int> myPDEDescriptionPort::getBoundaries() 
{
  return com->boundaries;
}
 
SSIDL::array1<int> myPDEDescriptionPort::getDirichletNodes()
{
  return com->dirichletNodes;
}

SSIDL::array1<double> myPDEDescriptionPort::getDirichletValues()
{
  return com->dirichletValues;
}






