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
 *  plodaer.cc: CCA Component Loader main file
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 */

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/spec/cca_sidl.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <Core/Thread/Thread.h>
#include <SCIRun/SCIRunLoader.h>
#include <iostream>
#include <sys/stat.h>
#include <mpi.h>

using namespace std;
using namespace SCIRun;
using namespace sci::cca;
#define VERSION "2.0.0" // this needs to be synced with the contents of
                        // SCIRun/doc/edition.xml


int
main(int argc, char *argv[] )
{
 /*Loader is MPI enabled*/
 MPI_Init(&argc,&argv);
 int mpi_size, mpi_rank;
 MPI_Comm_size(MPI_COMM_WORLD,&mpi_size);
 MPI_Comm_rank(MPI_COMM_WORLD,&mpi_rank);


  try {
    PIDL::initialize(mpi_rank, mpi_size);
  } 
  catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }

  //ifstream f("framework.url");
  //std::string s;
  //f>>s;
  //f.close();

  std::string loaderName=argv[1];
  std::string frameworkURL=argv[2];
  try {
    sci::cca::Loader::pointer ploader;
    SCIRunLoader *sl=new SCIRunLoader(loaderName, frameworkURL);
    ploader=sci::cca::Loader::pointer(sl);
    ploader->addReference();
    ploader->addReference();
    ploader->addReference();
    ploader->addReference();
    cout<< "ploader started!"<<endl;
    cout<< "framwork url="<<frameworkURL<<endl;

    MPI_Comm_size(MPI_COMM_WORLD,&(sl->mpi_size));
    MPI_Comm_rank(MPI_COMM_WORLD,&(sl->mpi_rank));


    //Inform everyone else of my distribution
    //(this is in correspondence with the instantiate() call)
    
    Index* dr[1];
    dr[0] = new Index((sl->mpi_rank),(sl->mpi_rank)+1,1);  //first, last, stride
    MxNArrayRep* arrr = new MxNArrayRep(1,dr);
    sl->setCalleeDistribution("dURL",arrr);   //server is callee
   
    vector<URL> vURL;
    vURL.push_back(frameworkURL);
    
    Object::pointer obj=PIDL::objectFrom(vURL,1,0);
    //Object::pointer obj=PIDL::objectFrom(frameworkURL,1,0);
    if(obj.isNull()){
      cerr<<"Cannot get framework from url="<<frameworkURL<<endl;
      return 0;
    }
    sci::cca::AbstractFramework::pointer framework=pidl_cast<sci::cca::AbstractFramework::pointer>(obj);
  
    typedef char urlString[100] ;
    urlString s;
    strcpy(s, ploader->getURL().getString().c_str());

    urlString *buf;

    if(sl->mpi_rank==0){
      buf=new urlString[sl->mpi_size];
    }

    MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);
    
    if(sl->mpi_rank==0){
      SSIDL::array1< std::string> URLs;
      for(int i=0; i<sl->mpi_size; i++){
	string url(buf[i]);
	URLs.push_back(url);
	cerr<<"ploader URLs["<<i<<"]="<<url<<endl;
      }
      framework->registerLoader(loaderName, URLs);
      delete buf;
    }
  }catch(const MalformedURL& e) {
	cerr << "slaveTest.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
  }
  catch(const Exception& e) {
    cerr << "Caught exception:\n";
    cerr << e.message() << '\n';
    abort();
  } catch(...) {
    cerr << "Caught unexpected exception!\n";
    abort();
  }

  PIDL::serveObjects();
  PIDL::finalize();

  MPI_Finalize();
  return 0;
}
