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
 *  DTAddress.h defines the unique address of each data transmitter
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef CORE_CCA_COMM_DT_DTADDRESS_H
#define CORE_CCA_COMM_DT_DTADDRESS_H
#include <iostream>
namespace SCIRun {
  class DTPoint;

  class DTAddress{
  public:
    int port;
    long ip;
    bool operator<(const DTAddress &p) const{
      return (ip<p.ip) || (ip==p.ip && port<p.port);
    }

    bool operator==(const DTAddress &p) const{
      return ip==p.ip && port==p.port;
    }
  };


  class DTDestination{
  public:
    DTAddress to_addr;
    DTPoint *recver;

    DTDestination(){}

    DTDestination(DTPoint *recver, const DTAddress &to_addr){
      this->recver=recver;
      this->to_addr=to_addr;
    }

    void unset(){
      recver=NULL;
    }

    bool isSet(){
      return recver!=NULL;
    }

    bool operator<(const DTDestination &dest) const{
      return (to_addr<dest.to_addr) || (to_addr==dest.to_addr && recver<dest.recver); 
    }

    bool operator==(const DTDestination &dest) const{
      return (to_addr==dest.to_addr && recver==dest.recver);
    }

    void display(){
      std::cerr<<"ip/port/recver="<<to_addr.ip<<"/"<<to_addr.port<<"/"<<recver<<std::endl;
    }
  };

  class DTPacketID{
  public:
    DTDestination dest;
    DTAddress fr_addr;

    DTPacketID(){};

    DTPacketID(const DTDestination &dest, const DTAddress &fr_addr){
      this->dest=dest;
      this->fr_addr=fr_addr;
    }

    bool operator<(const DTPacketID &pid) const{
      return (dest<pid.dest) || (dest==pid.dest && fr_addr<pid.fr_addr); 
    }

    bool operator==(const DTPacketID &pid) const{
      return (dest==pid.dest && fr_addr==pid.fr_addr);
    }
  };


}// namespace SCIRun

#endif
