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

#include <sci_defs/mpi_defs.h>
#include <iostream>
#include <vector>
#include <sci_mpi.h>
#include <Core/CCA/PIDL/PIDL.h>

#include <Core/CCA/PIDL/MalformedURL.h>

#include <testprograms/Component/exceptiontest/PingThrow_impl.h>
#include <testprograms/Component/exceptiontest/PingThrow_sidl.h>
#include <Core/Thread/Time.h>

using std::cerr;
using std::cout;
using std::string;

using PingThrow_ns::PingThrow_impl;
using PingThrow_ns::OtherThrow_impl;
using PingThrow_ns::PingThrow;
using PingThrow_ns::OtherThrow;
using namespace SCIRun;

void t1(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank);
void t2(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank);
void t3(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank);
void t4(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank);
void t5(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank);

void usage(char* progname)
{
    cerr << "usage: " << progname << " [options]\n";
    cerr << "valid options are:\n";
    cerr << "  -server  - server process\n";
    cerr << "  -client URL  - client process\n";
    cerr << "\n";
    exit(1);
}

//All will except
void t1(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank) 
{
  cout << "Test 1...  ";
  try {
    pp->pingthrow(1);
  } catch(PingThrow_ns::PPException* p) {
    cout << "SUCCESS\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) cout << "****************************************\n";
    t2(pp,ot,rank);
    return;
  }
  cout << "FAILED1\n";
  exit(1);
}

//Regular impreciseness
void t2(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank) 
{
  cout << "Test 2...  ";
  try {
    pp->pingthrow(rank+1); //rank zero excepts
    while(1) pp->donone();
  } catch(PingThrow_ns::PPException* p) {
    cout << "SUCCESS\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) cout << "****************************************\n";
    t3(pp,ot,rank);
    return;
  }
  cout << "FAILED1\n";
  exit(1);
}

//getException basic
void t3(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank) 
{
  cout << "Test 3...  ";
  try {
    pp->pingthrow(rank+1); //rank zero excepts
    pp->getException();
  } catch(PingThrow_ns::PPException* p) {
    cout << "SUCCESS\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) cout << "****************************************\n";
    t4(pp,ot,rank);
    return;
  }
  cout << "FAILED1\n";
  exit(1);
}

//getException with multiple exceptions/multiple methods
void t4(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank) 
{
  cout << "Test 4...  ";
  try {
    pp->pingthrow(rank); //rank one excepts
    pp->pingthrow2(rank+1); //rank zero excepts
    pp->getException();
  } catch(PingThrow_ns::PPException* p) {
    cout << "SUCCESS\n";
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0) cout << "****************************************\n";
    t5(pp,ot,rank);
    return;
  } catch(...) {
    cout << "FAILED1\n";
    exit(1);
  }
  cout << "FAILED2\n";
  exit(1);
}

//getException with multiple exceptions/multiple methods
void t5(PingThrow::pointer& pp, OtherThrow::pointer& ot, int rank) 
{
  cout << "Test 5...  ";
  pp->pingthrow(rank); //rank one excepts
  //PIDL::finalize should catch this
}

int main(int argc, char* argv[])
{
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

	  OtherThrow::pointer ot;
	  pp->getOX(ot);

	  cout << "Starting test...\n";
	  t1(pp,ot,myrank);
 	}

    PIDL::serveObjects();
    PIDL::finalize();
    } catch(PingThrow_ns::PPException* p) {
        cerr << "PASSED\n";
	cout << "ALL TESTS SUCCESSFUL\n";
        return(0);
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
    cerr << "FAILED\n";
    return 0;
}

