
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

#include <Util/Assert.h>

namespace SCICore {
namespace Containers {

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

    T* operator->() const;
    T* get_rep() const;
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
T* Handle<T>::operator->() const
{
    ASSERT(rep != 0);
    return rep;
}

template<class T>
T* Handle<T>::get_rep() const
{
    return rep;
}

} // End namespace Containers
} // End namespace SCICore

//
// $Log$
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


