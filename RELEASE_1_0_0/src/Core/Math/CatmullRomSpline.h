/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/share/share.h>
#include <Core/Containers/Array1.h>

namespace SCIRun {


template<class T>
class SCICORESHARE CatmullRomSpline {
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

} // End namespace SCIRun

////////////////////////////////////////////////////////////
// Start of included CatmullRomSpline.cc

#include <Core/Util/Assert.h>

namespace SCIRun {

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

} // End namespace SCIRun


#endif /* SCI_Math_CatmullRomSpline_h */
