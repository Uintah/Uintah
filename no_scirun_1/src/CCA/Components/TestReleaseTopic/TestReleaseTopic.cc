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


#include<CCA/Components/TestReleaseTopic/TestReleaseTopic.h>
#include <Framework/TypeMap.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestReleaseTopic()
{
  return sci::cca::Component::pointer(new TestReleaseTopic());
}


TestReleaseTopic::TestReleaseTopic()
{
}

TestReleaseTopic::~TestReleaseTopic()
{
  services->removeProvidesPort("go");
}

void TestReleaseTopic::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestReleaseTopicgo *providesgp = new TestReleaseTopicgo();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(TestReleaseTopicgo::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);

}

int TestReleaseTopicgo::go()
{

  std::cout<<"Inside Go Function of TestReleaseTopic\n";
   sci::cca::ports::EventService::pointer ptr;
   try {
      sci::cca::Port::pointer pp = com->getServices()->getPort("cca.EventService");
      ptr = pidl_cast<sci::cca::ports::EventService::pointer>(pp);
      if(ptr.isNull())
	std::cout << "Pointer returned from getPort is Null!!!\n";
   }
    catch (const sci::cca::CCAException::pointer &e) {
      std::cout << e->getNote() << std::endl;
    return 1;
  }
   //Releasing a topic with empty topic name
   std::cout << "Test 1 : Releasing a topic with empty topic name\n";
   try
     {
       std::string emptyString("");
       ptr->releaseTopic(emptyString);
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in trying to release a topic with empty topic name: " << e->getNote() << std::endl;
     }
   //Release a non-existent topic  
   std::cout << "Test 2 : Release a non-existent topic\n";
   try
     {
       ptr->releaseTopic(std::string("randomString"));
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in trying to release a non-existent topic: " << e->getNote() << std::endl;
     }
   //Release an existent topic  
   std::cout << "Test 3 : Release an existent topic\n";
   try
     {
       ptr->createTopic(std::string("Test"));
       ptr->releaseTopic(std::string("Test"));
       std::cout << "Release Topic successful\n";
     }
   catch(const sci::cca::EventServiceException::pointer &e)
     {
       std::cout << "Exception in trying to release an existent topic: " << e->getNote() << std::endl;
     }
   com->getServices()->releasePort("cca.EventService");
  return 0;
}

