#ifndef GFXTOOLS_ARRAY_INCLUDED // -*- C++ -*-
#define GFXTOOLS_ARRAY_INCLUDED

#include <string.h>

template<class T>
class array {
protected:
    T *data;
    int len;
public:
    array() { data=NULL; len=0; }
    array(int l) { init(l); }
    ~array() { free(); }


    inline void init(int l);
    inline void free();
    inline void resize(int l);

    inline T& ref(int i);
    inline T& operator[](int i) { return data[i]; }
    inline T& operator()(int i) { return ref(i); }

    inline const T& ref(int i) const;
    inline const T& operator[](int i) const { return data[i]; }
    inline const T& operator()(int i) const { return ref(i); }


    inline int length() const { return len; }
    inline int maxLength() const { return len; }
};

template<class T>
inline void array<T>::init(int l)
{
    data = new T[l];
    len = l;
}

template<class T>
inline void array<T>::free()
{
    if( data )
    {
	delete[] data;
	data = NULL;
    }
}

template<class T>
inline T& array<T>::ref(int i)
{
#ifdef SAFETY
    assert( data );
    assert( i>=0 && i<len );
#endif
    return data[i];
}

template<class T>
inline const T& array<T>::ref(int i) const
{
#ifdef SAFETY
    assert( data );
    assert( i>=0 && i<len );
#endif
    return data[i];
}

template<class T>
inline void array<T>::resize(int l)
{
    T *old = data;
    data = new T[l];
    data = (T *)memcpy(data,old,MIN(len,l)*sizeof(T));
    len = l;
    delete[] old;
}

// GFXTOOLS_ARRAY_INCLUDED
#endif
