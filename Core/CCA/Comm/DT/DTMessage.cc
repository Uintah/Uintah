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
 *  DTMessage.cc defines the message structure used in the data transmitter
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */



#include <iostream>
#include <string.h>
#include <Core/CCA/Comm/DT/DTMessage.h>
#include <Core/CCA/Comm/DT/DTAddress.h>
using namespace SCIRun;
using namespace std;


#define DISPLAY_MSG          

DTMessage::~DTMessage(){
  if(autofree && buf!=NULL) delete []buf;
}


void 
DTMessage::display(){
#ifdef DISPLAY_MSG      
  //char *str=new char[length];
  //strncpy(str, buf+sizeof(int), length-sizeof(int));
  //str[length-sizeof(int)]='\0';
  std::cerr<<"DTMessage:\n"
	   <<"\t recver="<<(long)recver<<"\n"
	   <<"\t sender="<<(long)sender<<"\n"
	   <<"\t fr_addr="<<fr_addr.ip<<"/"<<fr_addr.port<<"\n"
	   <<"\t to_addr="<<to_addr.ip<<"/"<<to_addr.port<<"\n"
	   <<"\t length="<<length<<"\n"
	   <<"\t offset="<<offset<<"\n"
    	   <<"\t buf(id)="<<*((int*)(this->buf))<<"\n";
#endif      
}

