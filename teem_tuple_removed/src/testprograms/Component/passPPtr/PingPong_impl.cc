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
 *  PingPong_impl.h: Test client for PIDL
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2003
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <testprograms/Component/passPPtr/PingPong_impl.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/NotFinished.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <mpi.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/PIDL/URL.h>
using namespace PingPong_ns;
using namespace std;
using namespace SCIRun;

void
PingPong_impl::setService(const Service::pointer &svc)
{
  int myrank;
  int mysize;
  MPI_Comm_size(MPI_COMM_WORLD,&mysize);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

  typedef char urlString[100] ;
  urlString s;
  UIPort::pointer port = UIPort::pointer(new Port_impl);
  strcpy(s, port->getURL().getString().c_str());
  
  urlString *buf;
  
  if(myrank==0){
    buf=new urlString[mysize];
  }
  
  MPI_Gather(  s, 100, MPI_CHAR,    buf, 100, MPI_CHAR,   0, MPI_COMM_WORLD);

  if(myrank==0){
    vector<URL> URLs;
    for(int i=0; i<mysize; i++){
      string url(buf[i]);
      URLs.push_back(url);
      cerr<<"port URLs["<<i<<"]="<<url<<endl;
    }
    Object::pointer obj=PIDL::objectFrom(URLs,mysize,myrank);
    Port::pointer port=pidl_cast<Port::pointer>(obj);
    svc->testPort(port);
    delete buf;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}


void 
Port_impl::ui(){
  int myrank;
  int mysize;
  MPI_Comm_size(MPI_COMM_WORLD,&mysize);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  if(myrank==0)  cerr<<"World Rank One";
  if(myrank==1)  cerr<<"World Rank Two";
}

void 
Service_impl::testPort(const Port::pointer &obj){
  cerr<<"Running pidl_cast in testPort...\n";
  UIPort::pointer port1=pidl_cast<UIPort::pointer>(obj);
  cerr<<"Done...\n";
  port1->ui();
}
