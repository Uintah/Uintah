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
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/MxNArrayRep.h>
#include <Core/CCA/PIDL/MalformedURL.h>
#include <testprograms/Component/mxnargtest/argtest_sidl.h>
#include <Core/Thread/Time.h>

using argtest::Server;
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

void init(SSIDL::array2<int>& a, int sta1, int fin1, int sta2, int fin2)
{
  a.resize(fin2-sta2,fin1-sta1);
//  std::cerr << "SIZE1 = " << a.size1() << " : SIZE2 = " << a.size2() << "\n";
  for(int j=sta2;j<fin2;j++)
    for(int i=sta1;i<fin1;i++) {
      a[j-sta2][i-sta1]=(j*100)+i;
//      std::cerr << "SLIDING IN a[" << j-sta2 << "][" << i-sta1 << "] = " << a[j-sta2][i-sta1] << "\n"; 
    }

  /*
  for(int j=sta2;j<=fin2;j++)
    for(int i=sta1;i<=fin1;i++) {
      std::cerr << "ARRAY sample arr[" << j-sta2 << "][" << i-sta1 << "] = " << a[j-sta2][i-sta1] << "\n";
    }

  std::cerr << "ARRAY sample arr[0][7] = " << a[0][7] << "\n";
  */  
}

bool test(SSIDL::array2<int>& a)
{
  bool success = true;
 
  for(unsigned int i=0;i<a.size1();i++)
    for(unsigned int j=0;j<a.size2();j++)
      if((a[i][j] % 100) != j)
	success = false;

  return success;
}

class Server_impl : public argtest::Server {
public:
  Server_impl();
  virtual ~Server_impl();
	
  virtual SSIDL::array2< int> return_arr();
  virtual void in_arr(const SSIDL::array2< int>& a);
  virtual void out_arr(SSIDL::array2< int>& a);
  virtual void inout_arr(SSIDL::array2< int>& a);
  virtual bool getSuccess();
private:
  bool success;
};

Server_impl::Server_impl()
{
  success = true;
}

Server_impl::~Server_impl()
{
}

SSIDL::array2< int> Server_impl::return_arr()
{
}

void Server_impl::in_arr(const SSIDL::array2< int>& a)
{
}

void Server_impl::out_arr(SSIDL::array2< int>& a)
{
  init(a,0,100,0,4); 
}

void Server_impl::inout_arr(SSIDL::array2< int>& a)
{
}

bool Server_impl::getSuccess() 
{
}

int main(int argc, char* argv[])
{

    int myrank = 0;
    int mysize = 0;
    int arrsize, sta1, fin1;

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
    	  SSIDL::array2<int> s_arr;
	  Server_impl* serv=new Server_impl;

          //Set up server's requirement of the distribution array 
	  Index** dr = new Index* [2]; 
	  dr[0] = new Index(myrank,99,mysize);
          dr[1] = new Index(myrank,2,mysize);
	  MxNArrayRep* arrr = new MxNArrayRep(2,dr);
	  serv->setCalleeDistribution("D",arrr);

	  cerr << "Waiting for MxN argtest connections...\n";
          /*Reduce all URLs and have root print them out*/
          typedef char urlString[100] ;
          urlString s;
	  strcpy(s, serv->getURL().getString().c_str());
	  urlString *buf;
          if(myrank==0){
	    buf=new urlString[mysize];
          }
	  MPI_Gather(s, 100, MPI_CHAR, buf, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
          if(myrank==0)
	    for(int i=0; i<mysize; i++) 
	      cerr << buf[i] << '\n';

	} else {
          SSIDL::array2<int> c_arr;
          //Creating a multiplexing proxy from all the URLs
	  Object::pointer obj=PIDL::objectFrom(server_urls,mysize,myrank);
	  Server::pointer serv=pidl_cast<Server::pointer>(obj);
	  if(serv.isNull()){
	    cerr << "serv_isnull\n";
	    abort();
	  }

	  //Set up the array and the timer  
	  double stime=Time::currentSeconds();
          arrsize = 100 / mysize;
          sta1 = myrank * arrsize;
          fin1 = (myrank * arrsize) + arrsize - 1;
          init(c_arr,sta1,fin1+1,0,2);

	  //Inform everyone else of my distribution
          //(this sends a message to all the callee objects)
          Index** dr = new Index* [2];
          dr[0] = new Index(sta1,fin1,1);
          dr[1] = new Index(0,1,1);
          MxNArrayRep* arrr = new MxNArrayRep(2,dr);
	  serv->setCallerDistribution("D",arrr); 

	  ::std::cerr << "INARR FINISHED\n";
          serv->in_arr(c_arr);
	  ::std::cerr << "INARR FINISHED -- OUTARR\n";
	  serv->out_arr(c_arr);
          ::std::cerr << "INARR FINISHED -- INOUTARR\n";
	  serv->inout_arr(c_arr);
          ::std::cerr << "INOUTARR FINISHED\n";

	  double dt=Time::currentSeconds()-stime;
	  cerr << reps << " reps in " << dt << " seconds\n";
	  double us=dt/reps*1000*1000;
	  cerr << us << " us/rep\n";
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
	abort();
    }
    PIDL::serveObjects();
    PIDL::finalize();
    MPI_Finalize();

    return 0;
}

