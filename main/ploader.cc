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
 *  plodaer.cc: CCA Component Loader main file
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <sci_defs/mpi_defs.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/DT/DataTransmitter.h>
#include <Core/CCA/Comm/PRMI.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/SCIRunLoader.h>
#include <iostream>
#include <sys/stat.h>
#include <sci_mpi.h>
#include <sci_defs/qt_defs.h>
#include <mpi.h>

using namespace std;
using namespace SCIRun;
using namespace sci::cca;
#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml

void
usage()
{
  std::cout << "CCA Parallel Component Loader" << std::endl;
  std::cout << "Usage: ploader [loader] [url]" << std::endl;
  std::cout << "       loader               : loader name" << std::endl;
  std::cout << "       url                  : framework url" << std::endl;
  exit(0);
}


int
main(int argc, char *argv[] )
{
    if (argc < 3) {
	usage();
    }
 /*Loader is MPI enabled*/
 MPI_Init(&argc,&argv);
 int mpi_size, mpi_rank;
 MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
 MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);


  try {
    PIDL::initialize(mpi_rank, mpi_size);
  } 
  catch(const Exception& e) {
    std::cerr << "Caught exception:\n";
    std::cerr << e.message() << '\n';
    abort();
  } catch(...) {
    std::cerr << "Caught unexpected exception!\n";
    abort();
  }

  std::string loaderName=argv[1];
  std::string frameworkURL=argv[2];
  try {
    sci::cca::Loader::pointer ploader;
    SCIRunLoader *sl=new SCIRunLoader(loaderName, frameworkURL);
    ploader=sci::cca::Loader::pointer(sl);
    
    //allowing the framework to access ploader by URL anytime
    ploader->addReference();

    std::cout << "ploader started." << std::endl;

    MPI_Comm_size(MPI_COMM_WORLD,&(sl->mpi_size));
    MPI_Comm_rank(MPI_COMM_WORLD,&(sl->mpi_rank));

    //start MPI lock manager
    PRMI::init();

    //Inform everyone else of my distribution
    //(this is in correspondence with the instantiate() call)
    Index* dr[1];
    dr[0] = new Index((sl->mpi_rank),(sl->mpi_rank)+1,1);  //first, last, stride
    MxNArrayRep* arrr = new MxNArrayRep(1,dr);
    sl->setCalleeDistribution("dURL",arrr);   //server is callee
   
    Object::pointer obj=PIDL::objectFrom(frameworkURL);
    if(obj.isNull()){
      std::cerr << "Cannot get framework from url=" << frameworkURL << std::endl;
      return 0;
    }

    sci::cca::AbstractFramework::pointer framework=pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
  
    typedef char urlString[100] ;
    urlString s;
    std::strcpy(s, ploader->getURL().getString().c_str());

#ifdef HAVE_MPI
    urlString *buf;
    if(sl->mpi_rank==0){
      buf=new urlString[sl->mpi_size];
    }
    MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);
    if(sl->mpi_rank==0){
      SSIDL::array1< std::string> URLs;
      for(int i=0; i<sl->mpi_size; i++){
	std::string url(buf[i]);
	URLs.push_back(url);
	std::cerr << "ploader URLs["<<i<<"]=" << url << std::endl;
      }
      framework->registerLoader(loaderName, URLs);
      delete buf;
    }
#else
    SSIDL::array1< std::string> URLs;
    std::string url(s);
    URLs.push_back(url);
    framework->registerLoader(loaderName, URLs);
#endif
  }catch(const MalformedURL& e) {
	std::cerr << "slaveTest.cc: Caught MalformedURL exception:\n";
	std::cerr << e.message() << '\n';
  }
  catch(const Exception& e) {
    std::cerr << "Caught exception:\n";
    std::cerr << e.message() << '\n';
    abort();
  } catch(...) {
    std::cerr << "Caught unexpected exception!\n";
    abort();
  }

  PIDL::serveObjects();
  PIDL::finalize();

  MPI_Finalize();
  return 0;
}
