
/*
 *  Ring.h: A static-length ring buffer
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/Ring.h>

template<class T> Ring<T>::Ring(int s)
: _head(0), _tail(0), _size(s)
{
    data.resize(s);
}

template<class T> Ring<T>::~Ring()
{
}
