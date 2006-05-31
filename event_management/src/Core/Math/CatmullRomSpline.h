/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <Core/Containers/Array1.h>

namespace SCIRun {


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
