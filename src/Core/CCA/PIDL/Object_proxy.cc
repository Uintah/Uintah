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
 *  Object_proxy.h
 *
 *  
 */

#include <Core/CCA/PIDL/Object_proxy.h>
#include <Core/CCA/PIDL/TypeInfo.h>
#include <Core/CCA/PIDL/URL.h>
#include <Core/CCA/PIDL/PIDL.h>
#include <Core/CCA/Comm/SocketMessage.h>
#include <unistd.h>
#include <iostream>
#include <string>

using namespace std;
using namespace SCIRun;

Object_proxy::Object_proxy(const Reference& ref)
    : ProxyBase(ref)
{
  rm.localSize = 1;
  rm.localRank = 0;
}

Object_proxy::Object_proxy(const URL& url)
{
  Reference *ref=new Reference();
  ref->chan->openConnection(url);
  ref->d_vtable_base=TypeInfo::vtable_methods_start;
  rm.insertReference(ref);

  rm.localSize = 1;
  rm.localRank = 0;
  //  rm.intracomm = NULL;
  //TODO: need verify this statement
  rm.s_lSize = 1;
}

Object_proxy::Object_proxy(const int urlc, const URL urlv[], int mysize, int myrank)
{
  for(int i=0; i < urlc; i++) {
    Reference *ref = new Reference();
    ref->chan->openConnection(urlv[i]);
    ref->d_vtable_base=TypeInfo::vtable_methods_start;
    rm.insertReference(ref);
  }
  rm.localSize = mysize;
  rm.s_lSize = mysize;
  rm.localRank = myrank;

//   if(mysize > 1)
//     rm.intracomm = PIDL::getIntraComm();
//   else
//     rm.intracomm = NULL;
}

Object_proxy::Object_proxy(const std::vector<URL>& urlv, int mysize, int myrank)
{
  std::vector<URL>::const_iterator iter = urlv.begin();
  for(unsigned int i=0; i < urlv.size(); i++, iter++) {
    Reference *ref = new Reference();
    ref->chan->openConnection(*iter);
    ref->d_vtable_base=TypeInfo::vtable_methods_start;
    rm.insertReference(ref);
  }
  rm.localSize = mysize;
  rm.s_lSize = mysize;
  rm.localRank = myrank;

//   if(mysize > 1)
//     rm.intracomm = PIDL::getIntraComm();
//   else
//     rm.intracomm = NULL;
}

Object_proxy::Object_proxy(const std::vector<Object::pointer>& pxy, int mysize, int myrank)
{
  std::vector<Object::pointer>::const_iterator iter = pxy.begin();
  for(unsigned int i=0; i < pxy.size(); i++, iter++) {
    ProxyBase* pbase = dynamic_cast<ProxyBase* >((*iter).getPointer());
    if(!pbase) continue;
    refList* refL = pbase->_proxyGetReferenceMgr()->getAllReferences();
    for(refList::const_iterator riter = refL->begin(); riter!=refL->end(); riter++){
      Reference *ref = (*riter)->clone();
      Message *message=ref->chan->getMessage();
      message->createMessage();
      message->sendMessage(SocketEpChannel::ADD_REFERENCE);
      rm.insertReference(ref);
    }
  }

  rm.localSize = mysize;
  rm.s_lSize = mysize;
  rm.localRank = myrank;

//   if(mysize > 1)
//     rm.intracomm = PIDL::getIntraComm();
//   else
//     rm.intracomm = NULL;
}


Object_proxy::~Object_proxy()
{
}















