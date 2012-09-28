/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#ifndef UINTAH_GRID_CONSTHANDLE_H
#define UINTAH_GRID_CONSTHANDLE_H

#include <Core/Util/Handle.h>
#include <Core/Util/Assert.h>

namespace Uintah {

/**************************************

CLASS
   Handle
   
   Short description...

GENERAL INFORMATION

   Handle.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Handle

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T> class constHandle {
  public:
    constHandle();
    constHandle(const T*);
    constHandle(const constHandle<T>&);    
    constHandle(const Handle<T>&);

    constHandle<T>& operator=(const constHandle<T>& copy)
    { return operator=(copy.d_rep); }
    constHandle<T>& operator=(const Handle<T>& copy)
    { return operator=(copy.get_rep()); }
    constHandle<T>& operator=(const T*);
    
    ~constHandle();
    
    void detach();
    
    inline const T* operator->() const {
      ASSERT(d_rep != 0);
      return d_rep;
    }
    inline const T* get_rep() const {
      return d_rep;
    }
    inline operator bool() const {
      return d_rep != 0;
    }
    inline bool operator == (const constHandle<T>& a) const {
      return a.d_rep == d_rep;
    }
    inline bool operator != (const constHandle<T>& a) const {
      return a.d_rep != d_rep;
    }
    inline bool operator == (const T* a) const {
      return a == d_rep;
    }
    inline bool operator != (const T* a) const {
      return a != d_rep;
    }
    inline bool operator == (int a) const {
      ASSERT(a == 0);
      return d_rep == 0;
    }
    inline bool operator != (int a) const {
      ASSERT(a == 0);
      return d_rep != 0;
    }
  private:
    const T* d_rep;
  };
  
  template<class T>
  constHandle<T>::constHandle()
    : d_rep(0)
  {
  }
  
  template<class T>
  constHandle<T>::constHandle(const T* rep)
    : d_rep(rep)
  {
    if(d_rep){
      d_rep->addReference();
    }
  }
  
  template<class T>
  constHandle<T>::constHandle(const constHandle<T>& copy)
    : d_rep(copy.d_rep)
  {
    if(d_rep){
      d_rep->addReference();
    }
  }

  template<class T>
  constHandle<T>::constHandle(const Handle<T>& copy)
    : d_rep(copy.get_rep())
  {
    if(d_rep){
      d_rep->addReference();
    }
  }
  
  template<class T>
  constHandle<T>& constHandle<T>::operator=(const T* copy)
  {
    if (d_rep != copy) {
      if(d_rep){
	if(d_rep->removeReference())
	  delete d_rep;
      }
      d_rep=copy;
      if(d_rep){
	d_rep->addReference();
      }
    }
    return *this;
  }
  
  template<class T>
  constHandle<T>::~constHandle()
  {
    if(d_rep){
      if(d_rep->removeReference())
	delete d_rep;
    }
  }
  
  template<class T>
  void constHandle<T>::detach()
  {
    ASSERT(d_rep != 0);
    d_rep->lock.lock();
    if(d_rep->ref_cnt==1){
      d_rep->lock.unlock();
      return; // We have the only copy
    }
    const T* oldrep=d_rep;
    d_rep=oldrep->clone();
    oldrep->ref_cnt--;
    oldrep->lock.unlock();
    d_rep->ref_cnt++;
  }

} // End namespace Uintah

#endif
