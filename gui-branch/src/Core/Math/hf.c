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


#include <stdio.h>
#include <math.h>

#ifdef __linux
  #include <values.h>
  #define fsqrt(x) ((float)sqrt(x))
#endif

#include <Core/Math/hf.h>

#ifdef _WIN32
  #include <float.h>
  #define MAXFLOAT FLT_MAX
  #define fsqrt(x) ((float)sqrt(x))
#endif

#ifdef _WIN32
#include <float.h>
#define MAXFLOAT FLT_MAX
#define fsqrt(x) ((float)sqrt(x))
#endif

#define CNORM(dx, dy, xx, yy) \
    x=(yy)*(dx); \
    y=(xx)*(dy); \
    z=(xx)*(yy); \
    l=x*x+y*y+z*z; \
    l=1.0F/fsqrt(l); \
    x*=l; \
    y*=l; \
    z*=l; \
    cur[0]=x; \
    cur[1]=y; \
    cur[2]=z;

void hf_float_s6(float* data, int xres, int yres)
{
    int ix, iy;
    int stride=xres*6;
    int xres1=xres-1;
    int yres1=yres-1;
    /* y=0 */
    float* cur=data;
    float* next=data+stride;
    float x,y,z,l,dx,dy;
    float* rlast;
    float* rcur;
    float* rnext;
    float* last;
    dy=next[5]-cur[5];
    dx=cur[11]-cur[5];
    CNORM(dx, dy, 1.0F, 1.0F);
    next+=6;
    cur+=6;

    for(ix=1;ix<xres;ix++){
	dy=next[5]-cur[5];
	dx=cur[11]-cur[-1];
	CNORM(dx, dy, 2.0F, 1.0F);
	next+=6;
	cur+=6;
    }
    dy=next[5]-cur[5];
    dx=cur[5]-cur[-1];
    CNORM(dx, dy, 1.0F, 1.0F);

    rlast=data;
    rcur=rlast+stride;
    rnext=rcur+stride;
    for(iy=1;iy<yres1;iy++){
	float* last=rlast;
	float* cur=rcur;
	float* next=rnext;
	dy=next[5]-last[5];
	dx=cur[11]-cur[5];
	CNORM(dx, dy, 1.0F, 2.0F);
	next+=6;
	last+=6;
	cur+=6;

	for(ix=1;ix<xres1;ix++){
	    float dy=next[5]-last[5];
	    float dx=cur[11]-cur[-1];
	    CNORM(dx, dy, 2.0F, 2.0F);
	    last+=6;
	    cur+=6;
	    next+=6;
	}
	dy=next[5]-last[5];
	dx=cur[5]-cur[-1];
	CNORM(dx, dy, 1.0F, 2.0F);

	rnext+=stride;
	rcur+=stride;
	rlast+=stride;
    }
    last=rlast;
    cur=rcur;
    dy=cur[5]-last[5];
    dx=cur[11]-cur[5];
    CNORM(dx, dy, 1.0F, 1.0F);
    cur+=6;
    last+=6;
    for(ix=1;ix<xres;ix++){
	dy=cur[5]-last[5];
	dx=cur[11]-cur[-1];
	CNORM(dx, dy, 2.0F, 1.0F);
	cur+=6;
	last+=6;
    }
    dy=cur[5]-last[5];
    dx=cur[5]-cur[-1];
    CNORM(dx, dy, 1.0F, 1.0F);
}

void hf_minmax_float_s6(float* data, int xres, int yres,
			float* pmin, float* pmax)
{
    int x,y;
    float* p=data;
    float min=MAXFLOAT;
    float max=-MAXFLOAT;
    for(y=0;y<yres;y++){
	for(x=0;x<xres;x++){
	    float f=p[5];
	    if(f<min)
		min=f;
	    if(f>max)
		max=f;
	    p+=6;
	}
    }
    *pmin=min;
    *pmax=max;
}
