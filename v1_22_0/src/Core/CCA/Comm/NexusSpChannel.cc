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


#include "NexusSpChannel.h"
#include <Core/CCA/Comm/NexusSpMessage.h>
#include <Core/CCA/Comm/CommError.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <iostream>
using namespace SCIRun;
using namespace std;
 
void NexusSpChannel::printDebug(const string& d) {
  cout << d << endl;
}

NexusSpChannel::NexusSpChannel() {
  //  if (kDEBUG) printDebug("NexusSpChannel::NexusSpChannel()");

  globus_nexus_startpoint_set_null(&d_sp);
}

NexusSpChannel::~NexusSpChannel() { 
  if (kDEBUG) printDebug("NexusSpChannel::~NexusSpChannel()");

  if (&d_sp != NULL) {
    if(int gerr=globus_nexus_startpoint_destroy_and_notify(&d_sp)){
      throw CommError("nexus_startpoint_destroy_and_notify", gerr);
    }
  }
}

SpChannel* NexusSpChannel::SPFactory(bool deep) {
  if (kDEBUG) printDebug("NexusSpChannel::SpFactory()");

  NexusSpChannel* new_sp = new NexusSpChannel();
  if (deep) {
    if(!globus_nexus_startpoint_is_null(&d_sp)) { 
      if( int gerr = globus_nexus_startpoint_copy
	  (&(new_sp->d_sp), const_cast<globus_nexus_startpoint_t*>(&d_sp)) ) {
	throw CommError("startpoint_copy", gerr);
      }
    }
  }
  else {
    new_sp->d_sp = d_sp;
  }
  return new_sp;
}

void NexusSpChannel::openConnection(const URL& url) {
  if(kDEBUG) {
    string s1("NexusSpChannel::openConnection() ");
    s1 += url.getString();
    printDebug(s1);
  }

  std::string s(url.getString());
  char* str=const_cast<char*>(s.c_str());
  if(int gerr=globus_nexus_attach(str, &d_sp)){
    throw CommError("nexus_attach", gerr);
  }
}

void NexusSpChannel::closeConnection() { 
  if (kDEBUG) printDebug("NexusSpChannel::closeConnection()");

  int size=0;
  globus_nexus_buffer_t buffer;

  if(int gerr=globus_nexus_buffer_init(&buffer, size, 0)) {
    throw CommError("buffer_init", gerr);
  }

  //Send the message
  int handler=TypeInfo::vtable_deleteReference_handler;
  if(int gerr=globus_nexus_send_rsr(&buffer, &d_sp,
  				    handler, GLOBUS_TRUE, GLOBUS_FALSE)) {
    throw CommError("ProxyBase: send_rsr", gerr);
  }
  //No reply is sent for this
  if(int gerr=globus_nexus_startpoint_destroy_and_notify(&d_sp)){
    throw CommError("nexus_startpoint_destroy_and_notify", gerr);
  }
}

Message* NexusSpChannel::getMessage() {
  if (kDEBUG) printDebug("NexusSpChannel::getMessage()");
  return (new NexusSpMessage(&d_sp));
}  






