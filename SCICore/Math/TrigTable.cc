
/*
 *  TrigTable.cc: Faster ways to do trig...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Math/TrigTable.h>
#include <SCICore/Math/Trig.h>

SinCosTable::SinCosTable(int n, double min, double max, double scale)
{
    sindata=new double[n];
    cosdata=new double[n];
    double d=max-min;
    for(int i=0;i<n;i++){
	double th=d*double(i)/double(n-1)+min;
	sindata[i]=Sin(th)*scale;
	cosdata[i]=Cos(th)*scale;
    }
}

SinCosTable::~SinCosTable()
{
    delete[] sindata;
    delete[] cosdata;
}

double SinCosTable::sin(int i) const
{
    return sindata[i];
}

double SinCosTable::cos(int i) const
{
    return cosdata[i];
}

