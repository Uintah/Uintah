
/*
 *  array.h: CIA array classes
 *  $Id$
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef Component_CIA_array_h
#define Component_CIA_array_h

#include <vector>

namespace CIA {
    template<class T> class array1 : public std::vector<T> {
    public:
	array1() : std::vector<T>(0) {}
	array1(unsigned long s) : std::vector<T>(s) {}
    };
}

#endif

//
// $Log$
// Revision 1.1  1999/09/28 08:19:48  sparker
// Implemented start of array class (only 1d currently)
// Implement string class (typedef to std::string)
// Updates to spec now that sidl support strings
//
//
