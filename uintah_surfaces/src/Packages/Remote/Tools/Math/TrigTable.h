/*
 *  TrigTable.h: Faster ways to do trig...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
*/

#ifndef _TrigTable_h
#define _TrigTable_h

#include <Packages/Remote/Tools/Math/MiscMath.h>

namespace Remote {
class SinCosTable
{
    double* sindata;
    double* cosdata;
    int n;
	
public:
	inline SinCosTable(int _n, double min, double max, double scale=1.0)
    {
		n = _n;
		sindata=new double[n];
		cosdata=new double[n];
		double d = max-min;
		for(int i=0;i<n;i++)
		{
			double th = d*double(i)/double(n-1) + min;
			sindata[i] = sin(th)*scale;
			cosdata[i] = cos(th)*scale;
		}
    }
	
	inline ~SinCosTable()
    {
		delete[] sindata;
		delete[] cosdata;
    }
	
	inline double Sin(int i) const
    {
		return sindata[i];
    }
	
	inline double Cos(int i) const
    {
		return cosdata[i];
    }
};

} // End namespace Remote


#endif
