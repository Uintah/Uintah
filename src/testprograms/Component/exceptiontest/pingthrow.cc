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
 *  pingthrow.cc
 *
 *  Written by:
 *   Kosta Damevski 
 *   Department of Computer Science
 *   University of Utah
 *   July 2003 
 *
 *  Copyright (C) 1999 U of U
 */

#include <iostream>
#include <vector>
#include <mpi.h>
#include <Core/CCA/PIDL/PIDL.h>

#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/exceptiontest/PingThrow_impl.h>
#include <testprograms/Component/exceptiontest/PingThrow_sidl.h>
#include <Core/Thread/Time.h>

using std::cerr;
using std::cout;

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

int main(int argc, char* argv[])
{
    using std::string;

    using PingThrow_ns::PingThrow_impl;
    using PingThrow_ns::PingThrow;

    try {
      PIDL::initialize();
      MPI_Init(&argc,&argv);
      
      int myrank = 0;
      int mysize = 1;
      bool client=false;
      bool server=false;
      std::vector <URL> server_urls;
      int reps=10;

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
	    PingThrow_impl::pointer pp(new PingThrow_impl);
	    pp->addReference();
	    if(myrank==0)
	      cerr << "Waiting for pingthrow connections...\n";

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
          Object::pointer obj=PIDL::objectFrom(server_urls,mysize,myrank);
	  cerr << "Object_from completed\n";

	  PingThrow::pointer pp=pidl_cast<PingThrow::pointer>(obj);
	  cerr << "pidl_case completed\n";
	  if(pp.isNull()){
	    cerr << "pp_isnull\n";
	    abort();
	  }
	  double stime=Time::currentSeconds();
	  for(int z=0; z<10; z++) { 
            ::std::cout << "Calling from node " << myrank << " for the " << z+1 << " time\n";
       	    int j=pp->pingthrow(13);
            //if(z==0) pp->getException();
	  }
          double dt=Time::currentSeconds()-stime;
	  cerr << "3 reps in " << dt << " seconds\n";
	  double us=dt/reps*1000*1000;
	  cerr << us << " us/rep\n";
 	}
    } catch(PingThrow_ns::PPException* p) {
        cerr << "pingthrow.cc: Coaught ppexception\n";
        //exit(0);
    } catch(const MalformedURL& e) {
	cerr << "pingthrow.cc: Caught MalformedURL exception:\n";
	cerr << e.message() << '\n';
    } catch(const Exception& e) {
	cerr << "pingthrow.cc: Caught exception:\n";
	cerr << e.message() << '\n';
	abort();
    } catch(...) {
	cerr << "Caught unexpected exception!\n";
	abort();
    }
    PIDL::serveObjects();
    PIDL::finalize();
    return 0;
}

