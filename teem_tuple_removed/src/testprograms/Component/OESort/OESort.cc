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
 *  OESort.cc
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 U of U
 */

#include <iostream>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/OESort/OESort_impl.h>
#include <testprograms/Component/OESort/OESort_sidl.h>
#include <Core/Thread/Time.h>

#define ARRSIZE 1000

using namespace SCIRun;
using namespace std;

void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "\n";
    exit(1);
}


int main(int argc, char* argv[])
{
    using std::string;

    using OESort_ns::OESort_impl;
    using OESort_ns::OESort;


    int myrank = 0;
    int mysize = 0;

    try {
        PIDL::initialize();
        
        MPI_Init(&argc,&argv);

	bool server=false;

        MPI_Comm_size(MPI_COMM_WORLD,&mysize);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	string arg(argv[1]);
	if(arg == "-server"){
	  server=true;
	} else {
	  usage(argv[0]);
	}

	if(server) {
	  OESort_impl* pp=new OESort_impl;

	  if(mysize != 2) {
	    cerr << "ERROR -- This was meant to be executed by 2 parallel processes\n";
	  }

          int localsize = ARRSIZE / mysize;
          int sta = myrank * localsize;
          int fin = (myrank * localsize) + localsize;
          if (myrank == mysize-1) fin = ARRSIZE;

          //Set up server's requirement of the distribution array 
          Index** dr0 = new Index* [1];
          dr0[0] = new Index(sta,fin,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(1,dr0);
	  pp->setCalleeDistribution("X",arrr0);

          Index** dr1 = new Index* [1];
          dr1[0] = new Index(sta,fin,1);
          MxNArrayRep* arrr1 = new MxNArrayRep(1,dr1);
	  pp->setCalleeDistribution("Y",arrr1);

          Index** dr2 = new Index* [1];
          dr2[0] = new Index(sta,fin,1);
          MxNArrayRep* arrr2 = new MxNArrayRep(1,dr2);
	  pp->setCalleeDistribution("Z",arrr2);

          std::cerr << "setCalleeDistribution completed\n";

	  cerr << "Waiting for OESort connections...\n";
          /*Reduce all URLs and have root print them out*/
          typedef char urlString[100] ;
          urlString s;
	  strcpy(s, pp->getURL().getString().c_str());
	  urlString *buf;
          if(myrank==0){
	    buf=new urlString[mysize];
          }
	  MPI_Gather(s, 100, MPI_CHAR, buf, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
          if(myrank==0)
	    for(int i=0; i<mysize; i++) 
	      cerr << buf[i] << '\n';

	}

    } catch(const MalformedURL& e) {
	cerr << "OESort.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const Exception& e) {
	cerr << "OESort.cc: Caught exception:\n";
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

