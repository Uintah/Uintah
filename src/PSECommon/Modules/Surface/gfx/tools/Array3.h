#ifndef GFXTOOLS_ARRAY3_INCLUDED // -*- C++ -*-
#define GFXTOOLS_ARRAY3_INCLUDED

template<class T>
class array3
{
protected:
    T *data;
    int w, h, d;

public:
    array3() { data=NULL; w=h=d=0; }
    array3(int w, int h, int d) { init(w,h,d); }
    ~array3() { free(); }

    inline void init(int w, int h, int d);
    inline void free();


    inline T& ref(int i, int j, int k);
    inline T& operator()(int i,int j, int k) { return ref(i,j, k); }

    inline int width() const { return w; }
    inline int height() const { return h; }
    inline int depth() const { return d; }
};

template<class T>
inline void array3<T>::init(int width, int height, int depth)
{
    w = width;
    h = height;
    d = depth;
    data = new T[w*h*d];
}

template<class T>
inline void array3<T>::free()
{
    if( data )
    {
	delete[] data;
	data = NULL;
	w = h = d = 0;
    }

}

template<class T>
inline T& array3<T>::ref(int i, int j, int k)
{
#ifdef SAFETY
    assert( data );
    assert( i>=0 && i<w );
    assert( j>=0 && j<h );
    assert( k>=0 && k<d );
#endif
    return data[k*w*h + j*w + i];
}

// GFXTOOLS_ARRAY3_INCLUDED
#endif
