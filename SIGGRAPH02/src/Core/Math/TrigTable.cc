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

