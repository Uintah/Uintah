
/*
 *  MusilRNG.h: Musil random number generator
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef sci_Math_MusilRNG_h
#define sci_Math_MusilRNG_h 1

#include <share/share.h>

class SHARE MusilRNG {
	int d[16], n[16];
	int stab[2][32];
	int point;
	int d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12;
	int a1,b1;
	int a2,b2;
	int a3,b3;
	int a4,b4;
	int a5,b5;
	int a6,b6;
	int a7,b7;
	int a8,b8;
	int a9,b9;
	int a10,b10;
	int a11,b11;
	int a12,b12;
public:
	MusilRNG(int=0);
	double operator()();
};

#endif

