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
 *  pingpong.cc
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   July 1999
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <vector>
#include <string>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>

#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/pingpongArr/PingPong_impl.h>
#include <testprograms/Component/pingpongArr/PingPong_sidl.h>
#include <Core/Thread/Time.h>
#include <assert.h>

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

void init(SSIDL::array1<int>& a, int s, int f)
{
  a.resize(f-s);
  for(int i=s;i<f;i++)
    a[i-s]=i;
}

int main(int argc, char* argv[])
{
    using std::string;

    using PingPong_ns::PingPong_impl;
    using PingPong_ns::PingPong;

    SSIDL::array1<int> arr;

    int myrank = 0;
    int mysize = 1;
    int arrsize, sta, fin;

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
	  PingPong_impl* pp=new PingPong_impl;

          //Set up server's requirement of the distribution array 
	  Index* dr[1];  
	  dr[0] = CYCLIC(myrank,mysize,100);
	  MxNArrayRep* arrr = new MxNArrayRep(1,dr);
	  pp->setCalleeDistribution("D",arrr);
          std::cerr << "setCalleeDistribution completed\n";

	  cerr << "Waiting for pingpong connections...\n";
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

	} else {

          //Creating a multiplexing proxy from all the URLs
	  Object::pointer obj=PIDL::objectFrom(server_urls,mysize,myrank);
	  PingPong::pointer pp=pidl_cast<PingPong::pointer>(obj);
	  if(pp.isNull()){
	    cerr << "pp_isnull\n";
	    abort();
	  }

	  //Set up the array and the timer  
	  double stime=Time::currentSeconds();
          cerr << mysize << ", " << myrank << "\n";
          arrsize = 100 / mysize;
          sta = myrank * arrsize;
          fin = (myrank * arrsize) + arrsize - 1;
          cerr << arrsize << "::" << sta << " -- " << fin << "\n";
          init(arr,sta,fin+1);

	  //Inform everyone else of my distribution
          //(this sends a message to all the callee objects)
          Index* dr[1];
          dr[0] = BLOCK(myrank,mysize,100);
	  cout << "sta=" << sta << ", dr[0]->myfirst)=" << dr[0]->myfirst << "\n";
	  cout << "fin=" << fin << ", dr[0]->myfirst)=" << dr[0]->mylast << "\n";
          MxNArrayRep* arrr = new MxNArrayRep(1,dr);
	  pp->setCallerDistribution("D",arrr); 
	  std::cerr << "setCallerDistribution completed\n";

	  for(int i=0;i<reps;i++){
            int j=pp->pingpong(arr);
	    cerr << "sum of all array elements = " << j << "\n";
	  }
	  double dt=Time::currentSeconds()-stime;
	  cerr << reps << " reps in " << dt << " seconds\n";
	  //double us=dt/reps*1000*1000;
	  //cerr << us << " us/rep\n";
	}
    } catch(const MalformedURL& e) {
	cerr << "pingpong.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const Exception& e) {
	cerr << "pingpong.cc: Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
        getchar();
	abort();
    }
    PIDL::serveObjects();
    PIDL::finalize();
    MPI_Finalize();

    return 0;
}

