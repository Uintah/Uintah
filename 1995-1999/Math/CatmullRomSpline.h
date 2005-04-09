
/*
 *  CatmullRomSpline.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Math_CatmullRomSpline_h
#define SCI_Math_CatmullRomSpline_h

#include <Classlib/Array1.h>

template<class T>
class CatmullRomSpline {
   Array1<T> d;
   int nintervals;
   int nset;
   int mx;

public:
   CatmullRomSpline();
   CatmullRomSpline( const Array1<T>& );
   CatmullRomSpline( const int );
   CatmullRomSpline( const CatmullRomSpline<T>& );

   void setData( const Array1<T>& );
   void add( const T& );
   void insertData( const int, const T& );
   void removeData( const int );
   
   T operator()( double ) const; // 0-1
   T& operator[]( const int );
};

#endif /* SCI_Math_CatmullRomSpline_h */
