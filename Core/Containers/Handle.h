
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

#include <SCICore/Util/Assert.h>
#include <SCICore/Persistent/Persistent.h>

namespace SCICore {
namespace Containers {

using SCICore::PersistentSpace::Piostream;

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

template<class T>
class Handle;
template<class T>
void Pio(Piostream& stream, Handle<T>& data);

template<class T>
class Handle {
    T* rep;
public:
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

    //////////
    //Destroy the handle
    ~Handle();

    void detach();

    inline T* operator->() const {
	ASSERT(rep != 0);
	return rep;
    }

    inline T* get_rep() const {
	return rep;
    }

    friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream& stream, Handle<T>& data);

};

} // End namespace Containers
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included Handle.cc
//

namespace SCICore {
namespace Containers {

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

template<class T>
void Pio(Piostream& stream, Handle<T>& data)
{
    stream.begin_cheap_delim();
    PersistentSpace::Persistent* trep=data.rep;
    stream.io(trep, T::type_id);
    if(stream.reading()){
	data.rep=(T*)trep;
	if(data.rep)
	    data.rep->ref_cnt++;
    }
    stream.end_cheap_delim();
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
// Revision 1.3  2000/02/02 22:07:06  dmw
// Handle - added detach and Pio
// TrivialAllocator - fixed mis-allignment problem in alloc()
// Mesh - changed Nodes from LockingHandle to Handle so we won't run out
// 	of locks for semaphores when building big meshes
// Surface - just had to change the forward declaration of node
//
// Revision 1.2  1999/08/17 06:38:36  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:12  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 19:55:43  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:04:31  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:26  dav
// Import sources
//
//

#endif


