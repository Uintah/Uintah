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
 *  Handle.h: Smart Pointers..
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Containers_Handle_h
#define SCI_Containers_Handle_h 1

#include <Core/Util/Assert.h>
#ifndef SCI_NOPERSISTENT
#include <Core/Persistent/Persistent.h>
#endif

namespace SCIRun {


/**************************************

 CLASS
 Handle

 KEYWORDS
 Handle

 DESCRIPTION

 Handle.h: Smart Pointers..

 Written by:
 Steven G. Parker
 Department of Computer Science
 University of Utah
 March 1994

 Copyright (C) 1994 SCI Group

 PATTERNS

 WARNING
  
****************************************/

template<class T> class Handle;
#ifndef SCI_NOPERSISTENT
template<class T> void Pio(Piostream& stream, Handle<T>& data);
#endif

template<class T> class Handle {
  T* rep;
public:

  typedef T   value_type;
  typedef T * pointer_type;

  //////////
  //Create the handle, initializing with a null pointer.
  Handle();

  //////////
  //Attach the handle to a pointer
  Handle(T*);
    
  //////////
  //Copy a handle from another handle
  Handle(const Handle<T>&);

  //////////
  //Assign a handle from another handle.
  Handle<T>& operator=(const Handle<T>&);

  //////////
  //Assign a handle from a pointer.
  Handle<T>& operator=(T*);

  bool operator==(const Handle<T>& crep) const;
  bool operator!=(const Handle<T>& crep) const;

  //////////
  //Destroy the handle
  ~Handle();

  void detach();

  inline T* operator->() {
    ASSERT(rep != 0);
    return rep;
  }

  inline T* get_rep() {
    return rep;
  }

  inline const T* operator->() const {
    ASSERT(rep != 0);
    return rep;
  }

  inline const T* get_rep() const {
    return rep;
  }

};

} // End namespace SCIRun

////////////////////////////////////////////////////////////
// Start of included Handle.cc

namespace SCIRun {

template<class T>
Handle<T>::Handle()
: rep(0)
{
}

template<class T>
Handle<T>::Handle(T* rep)
: rep(rep)
{
    if(rep)rep->ref_cnt++;
}

template<class T>
Handle<T>::Handle(const Handle<T>& copy)
: rep(copy.rep)
{
    if(rep)rep->ref_cnt++;
}

template<class T>
Handle<T>& Handle<T>::operator=(const Handle<T>& copy)
{
    if(rep != copy.rep){
	if(rep && --rep->ref_cnt==0)
	    delete rep;
	rep=copy.rep;
	if(rep)rep->ref_cnt++;
    }
    return *this;
}

template<class T>
Handle<T>& Handle<T>::operator=(T* crep)
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
    rep=crep;
    if(rep)rep->ref_cnt++;
    return *this;
}

template<class T>
bool Handle<T>::operator==(const Handle<T>& crep) const
{
  return (get_rep() == crep.get_rep());
}

template<class T>
bool Handle<T>::operator!=(const Handle<T>& crep) const
{
  return (get_rep() != crep.get_rep());
}

template<class T>
Handle<T>::~Handle()
{
    if(rep && --rep->ref_cnt==0)
	delete rep;
}

template<class T>
void Handle<T>::detach()
{
    ASSERT(rep != 0);
    if(rep->ref_cnt==1){
	return; // We have the only copy
    }
    T* oldrep=rep;
    rep=oldrep->clone();
    oldrep->ref_cnt--;
    rep->ref_cnt++;
}

#ifndef SCI_NOPERSISTENT
template<class T>
void Pio(Piostream& stream, Handle<T>& data)
{
    stream.begin_cheap_delim();
    Persistent* trep=data.get_rep();
    stream.io(trep, T::type_id);
    if(stream.reading()){
	data =(T*)trep;
    }
    stream.end_cheap_delim();
}
#endif

} // End namespace SCIRun


#endif


