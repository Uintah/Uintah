/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  TrigTable.cc: Faster ways to do trig...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 */

#include <Core/Math/TrigTable.h>
#include <Core/Math/Trig.h>

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

