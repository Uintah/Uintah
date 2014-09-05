#ifndef GFXTOOLS_BUFFER_INCLUDED // -*- C++ -*-
#define GFXTOOLS_BUFFER_INCLUDED

/*
 *  Buffer.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <gfx/tools/Array.h>

template<class T>
class buffer : public array<T> {
protected:
    int fill;
public:
    buffer() { init(8); }
    buffer(int l) { init(l); }
    
    inline void init(int l) { array<T>::init(l); fill=0; }

    inline int add(const T& t);
    inline void reset();
    inline int find(const T&);
    inline T remove(int i);
    inline int addAll(const buffer<T>& buf);
    inline void removeDuplicates();

    inline int length() const { return fill; }
    inline int maxLength() const { return len; }
};


template<class T>
inline int buffer<T>::add(const T& t)
{
    if( fill == len )
	resize( len*2 );

    data[fill] = t;

    return fill++;
}

template<class T>
inline void buffer<T>::reset()
{
    fill = 0;
}

template<class T>
inline int buffer<T>::find(const T& t)
{
    for(int i=0;i<fill;i++)
	if( data[i] == t )
	    return i;

    return -1;
}

template<class T>
inline T buffer<T>::remove(int i)
{
#ifdef SAFETY
    assert( i>=0 );
    assert( i<fill );
#endif

    fill--;
    T temp = data[i];
    data[i] = data[fill];

    return temp;
}

template<class T>
inline int buffer<T>::addAll(const buffer<T>& buf)
{
    for(int i=0; i<buf.fill; i++)
	add(buf(i));

    return fill;
}

template<class T>
inline void buffer<T>::removeDuplicates()
{
    for(int i=0; i<fill; i++)
    {
	for(int j=i+1; j<fill; )
	{
	    if( data[j] == data[i] )
		remove(j);
	    else
		j++;
	}
    }
}

//
// $Log$
// Revision 1.1  1999/07/27 16:58:06  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:18  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//

#endif // GFXTOOLS_BUFFER_INCLUDED
