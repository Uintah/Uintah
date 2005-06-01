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
#include <sci_defs/mpi_defs.h> // For MPIPP_H
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
    Object::pointer obj=PIDL::objectFrom(URLs,1,myrank);
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
