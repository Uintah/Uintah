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

namespace SCIRun {
  class DTAddress{
  public:
    int port;
    long ip;
    bool operator<(const DTAddress &p) const{
      return (port<=p.port && ip<p.ip) || (port<p.port && ip<=p.ip); 
    }
    bool operator==(const DTAddress &p) const{
      return port==p.port && ip==p.ip;
    }
  };

}// namespace SCIRun

#endif
