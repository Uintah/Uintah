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
 *  DTMessageTag.cc: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   Jun 2003
 *
 *  Copyright (C) 1999 SCI Group
 */



#include <stdlib.h>
#include <string>
#include <sstream>
#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <Core/Thread/Time.h>
#include <sys/time.h>


#include <iostream>
#include <Core/CCA/Comm/CommError.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/ConditionVariable.h>
#include <Core/CCA/Comm/DT/DTMessageTag.h>
#include <Core/CCA/Comm/DT/DTThread.h>
#include <Core/CCA/Comm/DT/DTPoint.h>
#include <Core/CCA/Comm/DT/DTMessage.h>

using namespace SCIRun;
using namespace std;

DTMessageTag::DTMessageTag(){
  hi=0;
  lo=1;
}

DTMessageTag::DTMessageTag(unsigned int hi, unsigned int lo){
  this->hi=hi;
  this->lo=lo;
}

DTMessageTag::~DTMessageTag(){
}

bool 
DTMessageTag::operator<(const DTMessageTag &tag) const{
  return (hi<tag.hi) || (hi==tag.hi && lo<tag.lo); 
}

bool 
DTMessageTag::operator==(const DTMessageTag &tag) const{
  return (hi==tag.hi) && (lo==tag.lo); 
}

DTMessageTag
DTMessageTag:: nextTag(){
  counter_mutex.lock();
  if(++lo==0) hi++;
  DTMessageTag newTag=*this;
  counter_mutex.unlock();
  return newTag;
}


DTMessageTag
DTMessageTag:: defaultTag(){
  DTMessageTag newTag;
  newTag.hi=newTag.lo=0;
  return newTag;
}

Mutex 
DTMessageTag::counter_mutex("DTMessageTag mutex");
