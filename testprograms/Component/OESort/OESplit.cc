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
 *  OESplit.cc
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
#include <stdlib.h>
#include <algo.h>
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
    cerr << "  -client URL  - client process\n";
    cerr << "\n";
    exit(1);
}

void init(SSIDL::array1<int>& a, int s)
{
  a.resize(s);
  a[0] = 1;
  for(int i=1;i<s;i++) {
    a[i] = (rand() % 10000);
  }
}

int main(int argc, char* argv[])
{
    using std::string;

    using OESort_ns::OESplit_impl;
    using OESort_ns::OESplit;
    using OESort_ns::OESort;

    SSIDL::array1<int> arr;
   
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
                i++;
                for(;i<argc;i++){
                  string url_arg(argv[i]);
                  server_urls.push_back(url_arg);
                }
		if(server_urls.size() == 0) 
		  usage(argv[0]);
		server=true;
                break;
	    } else if(arg == "-client"){
		if(server)
		    usage(argv[0]);
                i++;
                for(;i<argc;i++){		
                  string url_arg(argv[i]);
		  server_urls.push_back(url_arg);
                }
		if(server_urls.size() == 0) 
		  usage(argv[0]);
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
	  //THIS OBJECT IS BOTH A CLIENT AND SERVER
	  OESplit_impl* split=new OESplit_impl;

	  if(mysize != 2) {
	    cerr << "ERROR -- This was meant to be executed by 2 parallel processes\n";
	  }

          int localsize = ARRSIZE / mysize;
          int sta = myrank * localsize;
          int fin = sta + localsize;
          if (myrank == mysize-1) fin = ARRSIZE;

          //Set up server's requirement of the distribution array 
          Index** dr10 = new Index* [1];
          dr10[0] = new Index(sta,fin,1);
          MxNArrayRep* arrr10 = new MxNArrayRep(1,dr10);
	  split->setCalleeDistribution("A",arrr10);

          //Set up server's requirement of the distribution array 
          Index** dr11 = new Index* [1];
          dr11[0] = new Index(myrank,ARRSIZE,2);
          MxNArrayRep* arrr11 = new MxNArrayRep(1,dr11);
	  split->setCalleeDistribution("B",arrr11);

          std::cerr << "setCalleeDistribution completed\n";

	  cerr << "Waiting for OESplit connections...\n";
          /*Reduce all URLs and have root print them out*/
          typedef char urlString[100] ;
          urlString s;
	  strcpy(s, split->getURL().getString().c_str());
	  urlString *buf;
          if(myrank==0){
	    buf=new urlString[mysize];
          }
	  MPI_Gather(s, 100, MPI_CHAR, buf, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
          if(myrank==0)
	    for(int i=0; i<mysize; i++) 
	      cerr << buf[i] << '\n';

          //Creating a multiplexing proxy from all the URLs
          Object::pointer* obj = new Object::pointer;
          (*obj)=PIDL::objectFrom(server_urls,mysize,myrank);
          split->ss=pidl_cast<OESort::pointer>(*obj);
          if((split->ss).isNull()){
            cerr << "ss_isnull\n";
            abort();
          }

	  //Set up clients's requirement of the distribution array 
          Index** dr0 = new Index* [1];
          dr0[0] = new Index(sta,fin,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(1,dr0);
	  (split->ss)->setCallerDistribution("X",arrr0);

          Index** dr1 = new Index* [1];
          dr1[0] = new Index(sta,fin,2);
          MxNArrayRep* arrr1 = new MxNArrayRep(1,dr1);
	  (split->ss)->setCallerDistribution("Y",arrr1);

          Index** dr2 = new Index* [1];
	  sta = abs(myrank-1) * localsize;
	  fin = sta + localsize;
          dr2[0] = new Index(sta+1,fin,2);
          MxNArrayRep* arrr2 = new MxNArrayRep(1,dr2);
	  (split->ss)->setCallerDistribution("Z",arrr2);

          std::cerr << "setCallerDistribution completed\n";
	  

	} else {
	  
          //Creating a multiplexing proxy from all the URLs
	  Object::pointer obj=PIDL::objectFrom(server_urls,mysize,myrank);
	  OESplit::pointer pp=pidl_cast<OESplit::pointer>(obj);
	  if(pp.isNull()){
	    cerr << "pp_isnull\n";
	    abort();
	  }

	  //Set up the array and the timer  
          init(arr,ARRSIZE);

	  //Inform everyone else of my distribution
          //(this sends a message to all the callee objects)
          Index** dr0 = new Index* [1];
          dr0[0] = new Index(0,ARRSIZE,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(1,dr0);
	  pp->setCallerDistribution("A",arrr0);

          Index** dr1 = new Index* [1];
          dr1[0] = new Index(0,ARRSIZE,1);
          MxNArrayRep* arrr1 = new MxNArrayRep(1,dr1);
	  pp->setCallerDistribution("B",arrr1);

	  std::cerr << "setCallerDistribution completed\n";
	  
          double stime=Time::currentSeconds();

	  /*Odd-Even merge sort start*/
	  pp->split(arr,arr);

	  /*Pairwise check of merged array:*/
	  for(unsigned int arri = 0; arri+1 < arr.size(); arri+=2)
	    if (arr[arri] > arr[arri+1]) {
	      int t = arr[arri];
	      arr[arri] = arr[arri+1];
	      arr[arri+1] = t;
	    }
	  
	  double dt=Time::currentSeconds()-stime;
	  cerr << reps << " reps in " << dt << " seconds\n";
	  double us=dt/reps*1000*1000;
	  cerr << us << " us/rep\n";
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











