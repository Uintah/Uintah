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
 *  LUFactor.cc
 *
 *  Written by:
 *   Kostadin Damevski
 *   Department of Computer Science
 *   University of Utah
 *   October, 2002
 *
 *  Copyright (C) 2002 U of U
 */

#include <stdlib.h>

#include <iostream>
#include <algorithm>

#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <Core/Thread/Time.h>

#include <testprograms/Component/LUFactor/LUFactor_impl.h>
#include <testprograms/Component/LUFactor/LUFactor_sidl.h>


#define SIZE 5               

using namespace std;
using namespace SCIRun;

void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "  -client URL  - client process\n";
    cerr << "\n";
    exit(1);
}

void generate(SSIDL::array2<double>& A){
  unsigned int i, j;
  
  for( i=0; i<SIZE; i++ )       
    for ( j=0; j<SIZE; j++ )
      A[i][j] = (rand() % 10);
     
}

int main(int argc, char* argv[])
{
    using LUFactor_ns::LUFactor_impl;
    using LUFactor_ns::LUFactor;

    SSIDL::array2<double> A;
    A.resize(SIZE,SIZE); 

    int myrank = 0;
    int mysize = 0;

    try {
        PIDL::initialize(argc,argv);
        
        MPI_Init(&argc,&argv);

	bool client=false;
	bool server=false;
	vector <URL> server_urls;
	int reps=1;

        MPI_Comm_size(MPI_COMM_WORLD,&mysize);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	for(int i=1;i<argc;i++){
	    string arg(argv[i]);
	    if(arg == "-server"){
		if(client)
		    usage(argv[0]);
		server=true;
	    } else if(arg == "-client"){
		if(server)
		    usage(argv[0]);
                i++;
                for(;i<argc;i++){		
                  string url_arg(argv[i]);
		  server_urls.push_back(url_arg);
                }
		client=true;
                break;
	    } else if(arg == "-reps"){
		if(++i>=argc)
		    usage(argv[0]);
		reps=atoi(argv[i]);
	    } else {
		usage(argv[0]);
	    }
	}
	if(!client && !server)
	    usage(argv[0]);

	if(server) {
	  LUFactor_impl* lu=new LUFactor_impl;
	  
          //Set up server's requirement of the distribution array 
          Index** dr0 = new Index* [2];
	  dr0[0] = new Index(myrank,SIZE,mysize,mysize);
          dr0[1] = new Index(0,SIZE,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(2,dr0);
	  lu->setCalleeDistribution("X",arrr0);
          std::cerr << "setCalleeDistribution completed\n";
          /*Reduce all URLs and have root print them out*/
          typedef char urlString[100] ;
          urlString s;
	  strcpy(s, lu->getURL().getString().c_str());
	  urlString *buf;
          if(myrank==0){
	    buf=new urlString[mysize];
          }
	  MPI_Gather(s, 100, MPI_CHAR, buf, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
          if(myrank==0)
	    for(int i=0; i<mysize; i++) 
	      cerr << buf[i] << '\n';

	} else {

          //Creating a multiplexing proxy from all the URLs
	  Object::pointer obj=PIDL::objectFrom(server_urls,mysize,myrank);
	  LUFactor::pointer jc=pidl_cast<LUFactor::pointer>(obj);
	  if(jc.isNull()){
	    cerr << "jc_isnull\n";
	    abort();
	  }

	  //Set up a random matrix 
	  generate(A);                  

	  //Inform everyone else of my distribution
          //(this sends a message to all the callee objects)
          Index** dr0 = new Index* [2];
	  dr0[0] = new Index(0,SIZE,1);
          dr0[1] = new Index(0,SIZE,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(2,dr0);
	  jc->setCallerDistribution("X",arrr0);
          std::cerr << "setCallerDistribution completed\n";

          double stime=Time::currentSeconds();

	  /*Solve heat equation on this array*/
	  jc->LUFactorize(A);
	  
	  double dt=Time::currentSeconds()-stime;
	  cerr << reps << " reps in " << dt << " seconds\n";
	  double us=dt/reps*1000*1000;
	  cerr << us << " us/rep\n";

	  cout << "\n";
	}
    } catch(const MalformedURL& e) {
	cerr << "LUFactor.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const Exception& e) {
	cerr << "LUFactor.cc: Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
    PIDL::serveObjects();

    MPI_Finalize();

    return 0;
}











