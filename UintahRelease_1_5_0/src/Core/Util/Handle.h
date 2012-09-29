/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_GRID_HANDLE_H
#define UINTAH_GRID_HANDLE_H

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
  

KEYWORDS
   Handle

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

  template<class T> class Handle {
  public:
    Handle();
    Handle(T*);
    Handle(const Handle<T>&);
    
    Handle<T>& operator=(const Handle<T>& copy) { return operator=(copy.d_rep); }
    Handle<T>& operator=(T*);
    
    ~Handle();
    
    void detach();
    
    inline const T* operator->() const {
      ASSERT(d_rep != 0);
      return d_rep;
    }
    inline T* operator->() {
      ASSERT(d_rep != 0);
      return d_rep;
    }
    inline T* get_rep() {
      return d_rep;
    }
    inline const T* get_rep() const {
      return d_rep;
    }
    inline operator bool() const {
      return d_rep != 0;
    }
    inline bool operator == (const Handle<T>& a) const {
      return a.d_rep == d_rep;
    }
    inline bool operator != (const Handle<T>& a) const {
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
      a=a;     // This quiets the MIPS compilers
      return d_rep == 0;
    }
    inline bool operator != (int a) const {
      ASSERT(a == 0);
      a=a;     // This quiets the MIPS compilers
      return d_rep != 0;
    }
  private:
    T* d_rep;
  }; // end class Handle
  
  template<class T>
  Handle<T>::Handle()
    : d_rep(0)
  {
  }
  
  template<class T>
  Handle<T>::Handle(T* rep)
    : d_rep(rep)
  {
    if(d_rep){
      d_rep->addReference();
    }
  }
  
  template<class T>
  Handle<T>::Handle(const Handle<T>& copy)
    : d_rep(copy.d_rep)
  {
    if(d_rep){
      d_rep->addReference();
    }
  }
  
  template<class T>
  Handle<T>& Handle<T>::operator=(T* copy)
  {
    if(d_rep != copy){    
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
  Handle<T>::~Handle()
  {
    if(d_rep) {
      if(d_rep->removeReference()) {
        delete d_rep;
      }
    }
  }
  
  template<class T>
  void Handle<T>::detach()
  {
    ASSERT(d_rep != 0);
    d_rep->lock.lock();
    if(d_rep->ref_cnt==1){
      d_rep->lock.unlock();
      return; // We have the only copy
    }
    T* oldrep=d_rep;
    d_rep=oldrep->clone();
    oldrep->ref_cnt--;
    oldrep->lock.unlock();
    d_rep->ref_cnt++;
  }

} // End namespace Uintah

#endif
