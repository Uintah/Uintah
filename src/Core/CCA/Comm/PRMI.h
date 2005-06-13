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
 *  PRMI.h
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Oct 2004
 *
 *  Copyright (C) 1999 SCI Group
 */


#ifndef CORE_CCA_COMM_PRMI_H
#define CORE_CCA_COMM_PRMI_H

#include <sci_defs/mpi_defs.h>
#include <iostream>
#include <vector>
#include <deque>
#include <map>
#include <string>
#include <sstream>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/Comm/DT/DTAddress.h>
#include <Core/CCA/Comm/DT/DTMessageTag.h>
#include <Core/CCA/Comm/SocketEpChannel.h>
#include <sci_defs/mpi_defs.h>
#include <sci_mpi.h>


namespace SCIRun {

  class ProxyID{
  public:
    int iid;  //first level ID
    int pid;  //second level ID
    ProxyID(){
      iid=pid=0;
    }

    ProxyID(int iid, int pid){
      this->iid=iid;
      this->pid=pid;
    }

    bool isNull(){
      return iid==0 && pid==0;
    }

    bool operator==(const ProxyID &o)const{
      return iid==o.iid && pid==o.pid;
    }

    bool operator<(const ProxyID &o)const{
      return iid<o.iid || (iid==o.iid && pid<o.pid);
    }

    std::string str(){
      ::std::ostringstream s;
      s<<'|'<<iid<<'|'<<pid<<'|';
      return s.str();
    }

    ProxyID next(){
      return ProxyID(iid, pid+1);
    }

    ProxyID next1st(){
      return ProxyID(iid+1, pid);
    }
  };

  class PRMI{
  public:
    class lockid{
    public:
      bool operator!=(const lockid& o)const{
	return !(invID==o.invID && seq==o.seq);
      }
      bool operator<(const lockid& o)const{
	return invID<o.invID || (invID==o.invID && seq<o.seq);
      }
      std::string str(){
	std::ostringstream s;
	s<<invID.str()<<seq<<'|';
	return s.str();
      }

      ProxyID invID;
      int seq;
    };
    class states{
    public:
      ProxyID invID;
      int lockSeq;
      ProxyID nextID; 
      //...
    };

    static void init(int rank, int size);
    static void lock();
    static void unlock();

    static void internal_lock();
    static void internal_unlock();

    //#ifdef HAVE_MPI
    //duplicate MPI_COMM_WORLD
    //    static int getComm(MPI_Comm *);
    //#endif

    static void order_service(DTMessage *dtmsg);
    static void lock_service(DTMessage *dtmsg);
    static void addStat(states *);
    static void delStat();
    static states* getStat();
    static void setInvID(ProxyID);
    static ProxyID getInvID();
    //set next available UUID
    static void setProxyID(ProxyID nextID);
    //get next available UUID (nextID automatically updates)
     static ProxyID getProxyID();
    //peek next available UUID 
     static ProxyID peekProxyID();

    static int mpi_size;
    static int mpi_rank;
    //only root needs these
    static SocketEpChannel orderSvcEp;

    //everybody needs these
    static SocketEpChannel lockSvcEp;
    static DTPoint* orderSvc_ep;
    static DTAddress orderSvc_addr;

    //only root needs these
    static DTPoint** lockSvc_ep_list;
    static DTAddress  * lockSvc_addr_list;

    //#ifdef HAVE_MPI
    //    static MPI_Comm MPI_COMM_WORLD_Dup;
    //#endif

  private:


    static std::deque<PRMI::lockid> lockQ;

    static Mutex *lock_mutex;

    static Mutex *order_mutex;

    //used by lock service
    static std::map<Thread* ,states*> states_map;
    static std::map<PRMI::lockid ,Semaphore*> lock_sema_map;

    //used by order service
    static std::map<PRMI::lockid ,int> lock_req_map;

    //framework threads share the same state
    static states fwkstate;
  };
}//namespace SCIRun

#endif

