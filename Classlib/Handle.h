
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
class Handle {
    T* rep;
public:
    Handle();
    Handle(T*);
    Handle(const Handle<T>&);
    Handle<T>& operator=(const Handle<T>&);
    Handle<T>& operator=(T*);
    ~Handle();

    T* operator->() const;
    T* get_rep() const;
};

#endif
