/*
*  CatmullRomSpline.cc: 
*
*  Written by:
*   Steven G. Parker
*   Department of Computer Science
*   University of Utah
*   March 1994
*
*  Copyright (C) 1994 SCI Group
*/

#include <Packages/Remote/Tools/Math/CatmullRomSpline.h>
#include <Packages/Remote/Tools/Util/Assert.h>

namespace Remote {
template<class T>
CatmullRomSpline<T>::CatmullRomSpline()
: d(0), nset(0), nintervals(0), mx(0)
{
}

template<class T>
CatmullRomSpline<T>::CatmullRomSpline( const vector<T>& data )
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
CatmullRomSpline<T>::CatmullRomSpline( const CatmullRomSpline<T>& s )
: d(s.d), nset(s.nset), nintervals(s.nintervals), mx(s.mx)
{
}

template<class T>
void CatmullRomSpline<T>::setData( const vector<T>& data )
{
	d = data;
	nset = data.size();
	nintervals = nset-3;
	mx = nset-4;
}

template<class T>
void CatmullRomSpline<T>::add( const T& obj )
{
	d.push_back(obj);
	nset++;
	nintervals++;
	mx++;
}
#if 0
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
#endif

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
} // End namespace Remote


