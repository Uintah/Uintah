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
  
   Copyright (C) 2000 SCI Group

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
      Handle<T>& operator=(const Handle<T>&);
      Handle<T>& operator=(T*);
      ~Handle();
      
      void detach();
      
      inline const T* operator->() const {
	 //ASSERT(d_rep != 0);
	 return d_rep;
      }
      inline T* operator->() {
	 //ASSERT(d_rep != 0);
	 return d_rep;
      }
      inline T* get_rep() const {
	 return d_rep;
      }
      inline operator bool() const {
	 return d_rep != 0;
      }
   private:
      T* d_rep;
   };
   
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
      Handle<T>& Handle<T>::operator=(const Handle<T>& copy)
      {
	 if(d_rep != copy.d_rep){
	    if(d_rep){
	       if(d_rep->removeReference())
		  delete d_rep;
	    }
	    d_rep=copy.d_rep;
	    if(d_rep){
	       copy.d_rep->addReference();
	    }
	 }
	 return *this;
      }
   
   template<class T>
      Handle<T>& Handle<T>::operator=(T* copy)
      {
	 if(d_rep){
	    if(d_rep->removeReference())
	       delete d_rep;
	 }
	 d_rep=copy;
	 if(d_rep){
	    d_rep->addReference();
	 }
	 return *this;
      }
   
   template<class T>
      Handle<T>::~Handle()
      {
	 if(d_rep){
	    if(d_rep->removeReference())
	       delete d_rep;
	 }
      }
   
   template<class T>
      void Handle<T>::detach()
      {
	 //ASSERT(d_rep != 0);
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
