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

#include <vector>

namespace Remote {
using namespace std;
template<class T>
class CatmullRomSpline {
	vector<T> d;
	int nintervals;
	int nset;
	int mx;
	
public:
	CatmullRomSpline<T>();
	CatmullRomSpline<T>( const vector<T>& );
	CatmullRomSpline<T>( const int );
	CatmullRomSpline<T>( const CatmullRomSpline<T>& );
	
	void setData( const vector<T>& );
	void add( const T& );
	// void insertData( const int, const T& );
	// void removeData( const int );
	
	T operator()( double ) const; // 0-1
	T& operator[]( const int );
};

} // End namespace Remote


#endif /* SCI_Math_CatmullRomSpline_h */
