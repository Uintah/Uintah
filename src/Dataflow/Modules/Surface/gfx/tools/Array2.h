#ifndef GFXTOOLS_ARRAY2_INCLUDED // -*- C++ -*-
#define GFXTOOLS_ARRAY2_INCLUDED

/*
 *  Array2.h: ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

template<class T>
class array2 {
protected:
    T *data;
    int w, h;
public:
    array2() { data=NULL; w=h=0; }
    array2(int w, int h) { init(w,h); }
    ~array2() { free(); }

    inline void init(int w, int h);
    inline void free();

    inline T& ref(int i, int j);
    inline T& operator()(int i,int j) { return ref(i,j); }
    inline int width() const { return w; }
    inline int height() const { return h; }
};

template<class T>
inline void array2<T>::init(int width,int height)
{
    w = width;
    h = height;
    data = new T[w*h];
}

template<class T>
inline void array2<T>::free()
{
    if( data )
    {
	delete[] data;
	data = NULL;
    }
}

template<class T>
inline T& array2<T>::ref(int i, int j)
{
#ifdef SAFETY
    assert( data );
    assert( i>=0 && i<w );
    assert( j>=0 && j<h );
#endif
    return data[j*w + i];
}

//
// $Log$
// Revision 1.1  1999/07/27 16:58:06  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:17  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//


#endif  // GFXTOOLS_ARRAY2_INCLUDED
