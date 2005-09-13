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

    void display(){
      std::cerr<<"ip/port="<<ip<<"/"<<port<<std::endl;
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

  //deprecated, use DTMessageTag instead
/*  class DTPacketID{
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
*/

}// namespace SCIRun

#endif
