
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

#ifndef SCI_Classlib_Handle_h
#define SCI_Classlib_Handle_h 1

#include <Classlib/Assert.h>

template<class T>


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

#endif








