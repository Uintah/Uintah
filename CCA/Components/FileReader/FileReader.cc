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
 *  FileRaader.cc:
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
}

FileReader::~FileReader()
{
}

void FileReader::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  myPDEdescriptionPort::pointer pdep(new myPDEdescriptionPort);
  svc->addProvidesPort(pdep,"pde","sci.cca.ports.PDEdescriptionPort", props);
}

int
myPDEdescriptionPort::getPDEdescription(::SSIDL::array1<double> &nodes, 
					::SSIDL::array1<int> &boundaries,
					::SSIDL::array1<int> &dirichletNodes,
					::SSIDL::array1<double> &dirichletValues)
{
  QString fn = QFileDialog::getOpenFileName(
	    "./","PDE Description Files(*.pde)");
  if(fn.isNull()) return 1;
  ifstream is(fn);
  nodes.clear();
  boundaries.clear();
  dirichletNodes.clear();
  dirichletValues.clear();
  while(true){
    std::string name;
    is>>name;
    if(name=="node"){
      int cnt;
      is>>cnt;
      for(int i=0; i<cnt; i++){
	double x, y;
	is>>x>>y;
	nodes.push_back(x);
	nodes.push_back(y);
      }
    }
    else if(name=="boundary"){
      int cnt;
      is>>cnt;
      for(int i=0; i<cnt; i++){
	int index;
	is>>index;
	boundaries.push_back(index);
      }
    }
    else if(name=="dirichlet"){
      int cnt;
      is>>cnt;
      for(int i=0; i<cnt; i++){
	int index;
	is>>index;
	dirichletNodes.push_back(index);
      }
      for(int i=0; i<cnt; i++){
	double value;
	is>>value;
	dirichletValues.push_back(value);
      }
    }
    else if(name=="end") break;  

  }

  //  cerr<<nodes.size()<<endl;
  //  cerr<<boundaries.size()<<endl;
  //  cerr<<dirichletNodes.size()<<endl;
  //  cerr<<dirichletValues.size()<<endl;

  return 0;
}






