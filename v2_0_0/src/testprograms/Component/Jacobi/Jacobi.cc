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
 *  Jacobi.cc
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
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/Jacobi/Jacobi_impl.h>
#include <testprograms/Component/Jacobi/Jacobi_sidl.h>
#include <Core/Thread/Time.h>


/*****************************************************************************/
#define NYI    100                       /* Number of Y-Intervals            */
#define NXI    100                       /* Number of X-Intervals            */
#define Tinit    0                       /* Starting value                   */
/*****************************************************************************/

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

void initialize(SSIDL::array2<double>& t){
  unsigned int i, j;
  
  for( i=0; i<t.size1(); i++ )       
    for ( j=0; j<t.size2(); j++ )
      t[i][j] = Tinit;
}

int main(int argc, char* argv[])
{
    using std::string;

    using Jacobi_ns::Jacobi_impl;
    using Jacobi_ns::Jacobi;

    SSIDL::array2<double> t;
    t.resize(NXI,NYI); 

    int myrank = 0;
    int mysize = 0;

    try {
        PIDL::initialize();
        
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
	  Jacobi_impl* jc=new Jacobi_impl;

          int arrsize = NYI / mysize;
          int sta = myrank * arrsize;
          int fin = (myrank * arrsize) + arrsize;
	  if (myrank == mysize-1) fin = NYI;
	  
          //Set up server's requirement of the distribution array 
          Index** dr0 = new Index* [2];
	  dr0[0] = new Index(0,NXI,1);
          dr0[1] = new Index(sta,fin,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(2,dr0);
	  jc->setCalleeDistribution("X",arrr0);
          std::cerr << "setCalleeDistribution completed\n";

	  cerr << "Waiting for Jacobi connections...\n";
          /*Reduce all URLs and have root print them out*/
          typedef char urlString[100] ;
          urlString s;
	  strcpy(s, jc->getURL().getString().c_str());
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
	  Jacobi::pointer jc=pidl_cast<Jacobi::pointer>(obj);
	  if(jc.isNull()){
	    cerr << "jc_isnull\n";
	    abort();
	  }

	  //Set up the array 
	  initialize(t);                  

	  //Inform everyone else of my distribution
          //(this sends a message to all the callee objects)
          Index** dr0 = new Index* [2];
	  dr0[0] = new Index(0,NXI,1);
          dr0[1] = new Index(0,NYI,1);
          MxNArrayRep* arrr0 = new MxNArrayRep(2,dr0);
	  jc->setCallerDistribution("X",arrr0);
          std::cerr << "setCallerDistribution completed\n";

          double stime=Time::currentSeconds();

	  /*Solve heat equation on this array*/
	  jc->solveHeatEquation(t,300,100,0,0);

	  double dt=Time::currentSeconds()-stime;
	  cerr << reps << " reps in " << dt << " seconds\n";
	  double us=dt/reps*1000*1000;
	  cerr << us << " us/rep\n";
	  
	}
    } catch(const MalformedURL& e) {
	cerr << "Jacobi.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const Exception& e) {
	cerr << "Jacobi.cc: Caught exception:\n";
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


