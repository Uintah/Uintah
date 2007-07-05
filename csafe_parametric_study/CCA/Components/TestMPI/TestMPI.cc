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


#include<CCA/Components/TestMPI/TestMPI.h>
#include<Framework/TypeMap.h>
#include<mpi.h>

using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TestMPI()
{
  return sci::cca::Component::pointer(new TestMPI());
}


TestMPI::TestMPI()
{
}

TestMPI::~TestMPI()
{
  services->removeProvidesPort("go");
}

void TestMPI::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  TestMPIgo1 *providesgp = new TestMPIgo1();
  providesgp->setParent(this);
  sci::cca::TypeMap::pointer pProps0 = svc->createTypeMap();
  svc->addProvidesPort(sci::cca::ports::GoPort::pointer(providesgp), "go", "sci.cca.ports.GoPort", pProps0);
  //svc->registerUsesPort("MPIService", "cca.MPIService", pProps0);
}

int TestMPIgo1::go()
{
  try {

    std::cerr << "TestMPI started...\n";

    //someone has to call MPI_Init before using the service
    int argc = 0;
    char** argv = NULL;
    MPI_Init(&argc,&argv); 

    sci::cca::Services::pointer svc = com->getServices();
    if(svc.isNull()) {
      std::cerr << " bad services pointer\n";
      return -1;
    }
    sci::cca::Port::pointer sp = svc->getPort("cca.MPIService");
    if(sp.isNull()) {
      std::cerr << " found no port connected for MPIService\n";
      return -1;
    }

    sci::cca::ports::MPIService::pointer ms;
    ms = pidl_cast< sci::cca::ports::MPIService::pointer>(sp);
    if(ms.isNull()) {
      std::cerr << " found incorrect port type for MPIService\n";
      return -1;
    }
    std::cerr << "Got port to service.\n";
    
    long myComm = ms->getComm();
    if(myComm == 0) {
      std::cerr << " found 0 for MPIService\n";
      return -1;
    }
    MPI_Fint fmComm = (MPI_Fint)myComm;
    MPI_Comm fComm = MPI_Comm_f2c(fmComm);
    if(fComm == MPI_COMM_NULL) {
      std::cerr << " found MPI_COMM_NULL for MPIService\n";
      return -1;
    }
    
    int frank = 0;
    frank = MPI_Comm_rank(fComm, &frank);
    std::cerr << "found service rank = " << frank << "\n";
    
    ms->releaseComm(myComm);
    svc->releasePort("cca.MPIService");
    
    std::cerr << "TestMPI.go completed ok.\n";
    return 0;

  } catch ( sci::cca::CCAException::pointer& ex ) {
    std::string s = ex->getNote();
    std::cerr << "ERROR TestMPI.go FAILED. -- " << s << "\n";
    return -1;
  }

}

