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

double linalg_norm2(int rows, double* data)
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

