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
 *  LinAlg.h:  Tuned linear algebra routines
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <math.h>
#include <Core/Math/LinAlg.h>

double linalg_norm2(int rows, const double* data)
{
    double norm=0;
    int i;
    double sum0=0;
    double sum1=0;
    double sum2=0;
    double sum3=0;
    int r3=rows-3;
    for(i=0;i<r3;i+=4){
	double d0=data[i+0];
	double d1=data[i+1];
	double d2=data[i+2];
	double d3=data[i+3];
	sum0+=d0*d0;
	sum1+=d1*d1;
        sum2+=d2*d2;
	sum3+=d3*d3;
    }
    norm=(sum0+sum1)+(sum2+sum3);
    for(;i<rows;i++){
	double d=data[i];
	norm+=d*d;
    }
    return sqrt(norm);
}

void linalg_mult(int rows, double* res, double* a, double* b)
{
    int i=0;
    for(;i<rows;i++){
	res[i]=a[i]*b[i];
    }
}

void linalg_sub(int rows, double* res, double* a, double* b)
{
    int i=0;
    for(;i<rows;i++){
	res[i]=a[i]-b[i];
    }
}

void linalg_add(int rows, double* res, double* a, double* b)
{
    int i=0;
    for(;i<rows;i++){
	res[i]=a[i]+b[i];
    }
}

double linalg_dot(int rows, double* a, double* b)
{
    double dot=0;
    int i;
    double sum0=0;
    double sum1=0;
    double sum2=0;
    double sum3=0;
    int r3=rows-3;
    for(i=0;i<r3;i+=4){
	double a0=a[i+0];
	double a1=a[i+1];
	double a2=a[i+2];
	double a3=a[i+3];
	double b0=b[i+0];
	double b1=b[i+1];
	double b2=b[i+2];
	double b3=b[i+3];
	sum0+=a0*b0;
	sum1+=a1*b1;
        sum2+=a2*b2;
	sum3+=a3*b3;
    }
    dot=(sum0+sum1)+(sum2+sum3);
    for(;i<rows;i++){
	dot+=a[i]*b[i];
    }
    return dot;
}

void linalg_smadd(int rows, double* res, double s, double* a, double* b)
{
    int i=0;
    for(;i<rows;i++){
	res[i]=s*a[i]+b[i];
    }
}

void linalg_tridiag(int rows, LinAlg_TriDiagRow* data, double* c)
{
    int i;
    for(i=1;i<rows;i++){
	double factor=data[i][0]/data[i-1][1];

	data[i][1] -= factor*data[i-1][2];
	c[i] -= factor*c[i-1];
    }
    c[rows-1] = c[rows-1]/data[rows-1][1];
    for(i=rows-2;i>=0;i--){
	c[i] = (c[i]-data[i][2]*c[i+1])/data[i][1];
    }
}
