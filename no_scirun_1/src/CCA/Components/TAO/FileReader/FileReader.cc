/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


//
// FileReader.cc,v 1.3 2002/01/29 20:21:07 okellogg Exp
//
#include "FileReader.h"
#include <iostream>
#include <fstream>

#include <Core/Util/Environment.h>
#include <Framework/TypeMap.h>

ACE_RCSID(FileReader, FileReader, "FileReader.cc,v 1.3 2002/01/29 20:21:07 okellogg Exp")

FileReader::FileReader (CORBA::ORB_ptr orb)
  : orb_ (CORBA::ORB::_duplicate (orb))
{
}

FileReader* _main (int argc, char *argv[]);

extern "C" SCIRun::tao::Component* make_Tao_FileReader()
{
  return _main(0,NULL);
}

void FileReader::setServices(sci::cca::TaoServices::pointer services)
{
  sci::cca::TypeMap::pointer props(new SCIRun::TypeMap);
  services->addProvidesPort("hello", "corba.FileReader", props);
}

CORBA::Long 
FileReader::getPDEdescription (
			       Test::FileReader::double_array_out nodes ,
			       Test::FileReader::long_array_out boundaries,
			       Test::FileReader::long_array_out dirichletNodes,
			       Test::FileReader::double_array_out dirichletValues
			       ACE_ENV_ARG_DECL_WITH_DEFAULTS
			       )
  ACE_THROW_SPEC ((CORBA::SystemException ))
{
  nodes = new Test::FileReader::double_array;
  boundaries=new Test::FileReader::long_array;
  dirichletNodes=new Test::FileReader::long_array;
  dirichletValues=new Test::FileReader::double_array;


  std::string srcdir(SCIRun::sci_getenv("SCIRUN_SRCDIR"));
  ifstream is((srcdir + std::string("/CCA/Components/TAO/FileReader/L.pde")).c_str());


  while(true){
    std::string name;
    is>>name;
    if(name=="node"){
      unsigned int cnt;
      is>>cnt;
      nodes->length(cnt*2);
      for(unsigned int i=0; i<cnt; i++){
	double x, y;
	is>>x>>y;
	nodes[i+i]=x;
	nodes[i+i+1]=y;
      }
    }
    else if(name=="boundary"){
      unsigned  int cnt;
      is>>cnt;
      boundaries->length(cnt);

      for(unsigned int i=0; i<cnt; i++){
	int index;
	is>>index;
	boundaries[i]=index;
      }
    }
    else if(name=="dirichlet"){
      unsigned int cnt;
      is>>cnt;
      dirichletNodes->length(cnt);
      dirichletValues->length(cnt);

      for(unsigned int i=0; i<cnt; i++){
	int index;
	is>>index;
	dirichletNodes[i]=index;
      }
      for(unsigned int i=0; i<cnt; i++){
	double value;
	is>>value;
	dirichletValues[i]=value;
      }
    }
    else if(name=="end") break;  
  }

  return 0;

}

