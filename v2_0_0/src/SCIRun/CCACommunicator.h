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
 *  CCACommunicator.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 */

#ifndef SCIRun_CCACommunicator_h
#define SCIRun_CCACommunicator_h

#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>

#include <string>
using namespace std;

namespace SCIRun {

class SCIRunFramework;

struct Packet 
{
  char fromAddress[100];
  char type;  //reserved;
  char ccaFrameworkURL[256];      
};

class CCACommunicator:public Runnable{
 public:
  CCACommunicator(SCIRunFramework *framework,
		  const sci::cca::Services::pointer &svc);
  ~CCACommunicator(){}
  void run();

 protected:
  void readPacket(const Packet &pkt);
  Packet makePacket(int i);
  vector<string> ccaFrameworkURL; //framework URLs
  vector<string> ccaSiteList; //machine IP addresses
  SCIRunFramework *framework;
  sci::cca::Services::pointer services;
};


} //namespace SCIRun
#endif

 





