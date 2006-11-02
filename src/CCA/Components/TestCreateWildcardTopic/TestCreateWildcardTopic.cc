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


#include<CCA/Components/TestCreateWildcardTopic/TestCreateWildcardTopic.h>
#include <SCIRun/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestCreateWildcardTopic()
{
  return sci::cca::Component::pointer(new TestCreateWildcardTopic());
}


TestCreateWildcardTopic::TestCreateWildcardTopic()
{
}

TestCreateWildcardTopic::~TestCreateWildcardTopic()
{
  services->removeProvidesPort("go");
}

void TestCreateWildcardTopic::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestCreateWildcardTopicgo *providesgp0 = new TestCreateWildcardTopicgo();
  providesgp0->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestCreateWildcardTopicgo::pointer(providesgp0), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestCreateWildcardTopicgo::go()
{
  std::cout<<"Inside Go Function of TestCreateWildcardTopic\n";
   sci::cca::ports::EventService::pointer ptr;
   try {
      sci::cca::Port::pointer pp = com->getServices()->getPort("cca.EventService");
      ptr = pidl_cast<sci::cca::ports::EventService::pointer>(pp);
      if(ptr.isNull())
	std::cout << "Pointer is Null!!!\n";
   }
    catch (const sci::cca::CCAException::pointer &e) {
      std::cout << e->getNote() << std::endl;
    return 1;
    }
   //Try creating a WildcardTopic with some characters after '*'
   std::cout << "Test 1 : Try creating a WildcardTopic with some characters after '*'\n";
   try{
     sci::cca::WildcardTopic::pointer wildcardTopicPtr(ptr->createWildcardTopic("test.*.test"));
   }
   catch(const sci::cca::EventServiceException::pointer &e){
     std::cout << "Exception in creating a WildcardTopic: " << e->getNote() << std::endl; 
   }
   
   //Try creating a WildcardTopic with more than  '*'
   std::cout << "Test 2 : Try creating a WildcardTopic with more than  '*'\n";
   try{
     sci::cca::WildcardTopic::pointer wildcardTopicPtr(ptr->createWildcardTopic("test.*.test.*"));
   }
   catch(const sci::cca::EventServiceException::pointer &e){
     std::cout << "Exception in creating a WildcardTopic: " << e->getNote() << std::endl; 
   }
   return 0;
}

