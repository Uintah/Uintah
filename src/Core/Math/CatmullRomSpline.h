
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

#include <share/share.h>

#include <Containers/Array1.h>

namespace SCICore {
namespace Math {

using SCICore::Containers::Array1;

template<class T>
class SHARE CatmullRomSpline {
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

} // End namespace Math
} // End namespace SCICore

////////////////////////////////////////////////////////////
//
// Start of included CatmullRomSpline.cc
//

#include <Util/Assert.h>

namespace SCICore {
namespace Math {

template<class T>
CatmullRomSpline<T>::CatmullRomSpline()
: d(0), nset(0), nintervals(0), mx(0)
{
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline( const Array1<T>& data )
: d(data), nset(data.size()),
  nintervals(data.size()-3), mx(data.size()-4)
{
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline( const int n )
: d(n), nset(n), nintervals(n-3), mx(n-4)
{
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline( const CatmullRomSpline& s )
: d(s.d), nset(s.nset), nintervals(s.nintervals), mx(s.mx)
{
}

template<class T>
void CatmullRomSpline<T>::setData( const Array1<T>& data )
{
   d = data;
   nset = data.size();
   nintervals = nset-3;
   mx = nset-4;
}

template<class T>
void CatmullRomSpline<T>::add( const T& obj )
{
   d.add(obj);
   nset++;
   nintervals++;
   mx++;
}

template<class T>
void CatmullRomSpline<T>::insertData( const int idx, const T& obj )
{
   d.insert(idx, obj);
   nset++;
   nintervals++;
   mx++;
}

template<class T>
void CatmullRomSpline<T>::removeData( const int idx )
{
   d.remove(idx);
   nset--;
   nintervals--;
   mx--;
}

template<class T>
T CatmullRomSpline<T>::operator()( double x ) const
{
   ASSERT(nset >= 4);
   double xs(x*nintervals);
   int idx((int)xs);
   double t(xs-idx);
   if(idx<0){idx=0;t=0;}
   if(idx>mx){idx=mx;t=1;}
   double t2(t*t);
   double t3(t*t*t);
   
   return ((d[idx]*-1 + d[idx+1]*3  + d[idx+2]*-3 + d[idx+3])   *(t3*0.5)+
	   (d[idx]*2  + d[idx+1]*-5 + d[idx+2]*4  + d[idx+3]*-1)*(t2*0.5)+
	   (d[idx]*-1               + d[idx+2]                 )*(t*0.5)+
	   (            d[idx+1]                               ));
}

template<class T>
T& CatmullRomSpline<T>::operator[]( const int idx )
{
   return d[idx];
}

} // End namespace Math
} // End namespace SCICore

//
// $Log$
// Revision 1.1  1999/07/27 16:57:02  mcq
// Initial commit
//
// Revision 1.4  1999/07/01 16:44:22  moulding
// added SHARE to enable win32 shared libraries (dll's)
//
// Revision 1.3  1999/05/06 19:56:17  dav
// added back .h files
//
// Revision 1.1  1999/05/05 21:05:21  dav
// added SCICore .h files to /include directories
//
// Revision 1.1.1.1  1999/04/24 23:12:23  dav
// Import sources
//
//

#endif /* SCI_Math_CatmullRomSpline_h */
