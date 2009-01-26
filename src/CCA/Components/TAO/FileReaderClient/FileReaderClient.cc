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
#include "FileReaderClient.h"
#include <iostream>
#include <fstream>

#include <Framework/TypeMap.h>

const char *ior = "file://test.ior";

FileReaderClient::FileReaderClient(CORBA::ORB_ptr orb)
  : orb_ (CORBA::ORB::_duplicate (orb))
{
}

extern "C" SCIRun::tao::Component* make_Tao_FileReaderClient()
{
  int argc = 0;
  CORBA::ORB_var orb =
    CORBA::ORB_init(argc, NULL, "" ACE_ENV_ARG_PARAMETER);
  return new FileReaderClient(orb);
}

void FileReaderClient::setServices(sci::cca::TaoServices::pointer services)
{
  services->registerUsesPort("hello", "corba.FileReader", sci::cca::TypeMap::pointer(new SCIRun::TypeMap));
  services->addProvidesPort("go", "corba.Go", sci::cca::TypeMap::pointer(new SCIRun::TypeMap));
}

int FileReaderClient::go()
{
  CORBA::Object_var tmp =
  orb_->string_to_object(ior ACE_ENV_ARG_PARAMETER);
  ACE_TRY_CHECK;

  Test::FileReader_var hello =
   Test::FileReader::_narrow(tmp.in () ACE_ENV_ARG_PARAMETER);
  ACE_TRY_CHECK;

  if (CORBA::is_nil (hello.in ()))
  {
    ACE_ERROR_RETURN ((LM_DEBUG,
                      "Nil Test::FileReader reference <%s>\n",
                      ior),
                     1);
  }

  Test::FileReader::double_array_var nodes = new Test::FileReader::double_array;
  Test::FileReader::long_array_var boundaries=new Test::FileReader::long_array;
  Test::FileReader::long_array_var dirichletNodes=new Test::FileReader::long_array;
  Test::FileReader::double_array_var dirichletValues=new Test::FileReader::double_array;

  hello->getPDEdescription(nodes, boundaries, dirichletNodes, dirichletValues ACE_ENV_ARG_DECL_WITH_DEFAULTS);
  ACE_TRY_CHECK;

  for(unsigned int i=0; i< nodes->length(); i++){
    std::cout<<"nodes["<<i<<"]="<<nodes[i]<<std::endl;
  }

  orb_->destroy (ACE_ENV_SINGLE_ARG_PARAMETER);
  ACE_TRY_CHECK;
  return 0;
}
