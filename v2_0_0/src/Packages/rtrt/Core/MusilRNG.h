
#ifndef MUSILRNG_H
#define MUSILRNG_H 1

#include <math.h>

namespace rtrt {

class MusilRNG {
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
    unsigned int x1, x2;
    MusilRNG(int seed=0);
    inline double operator()() {

#if 1
	point=(point+20+stab[d1&1][a1+b1])&0x0f;
	d1=a1;a1=b1;b1=n[point];
	point=(point+20+stab[d2&1][a2+b2])&0x0f;
	d2=a2;a2=b2;b2=n[point];
	point=(point+20+stab[d3&1][a3+b3])&0x0f;
	d3=a3;a3=b3;b3=n[point];
	point=(point+20+stab[d4&1][a4+b4])&0x0f;
	d4=a4;a4=b4;b4=n[point];
	point=(point+20+stab[d5&1][a5+b5])&0x0f;
	d5=a5;a5=b5;b5=n[point];
	point=(point+20+stab[d6&1][a6+b6])&0x0f;
	d6=a6;a6=b6;b6=n[point];
	point=(point+20+stab[d7&1][a7+b7])&0x0f;
	d7=a7;a7=b7;b7=n[point];
	point=(point+20+stab[d8&1][a8+b8])&0x0f;
	d8=a8;a8=b8;b8=n[point];
	point=(point+20+stab[d9&1][a9+b9])&0x0f;
	d9=a9;a9=b9;b9=n[point];
	point=(point+20+stab[d10&1][a10+b10])&0x0f;
	d10=a10;a10=b10;b10=n[point];
	point=(point+20+stab[d11&1][a11+b11])&0x0f;
	d11=a11;a11=b11;b11=n[point];
	point=(point+20+stab[d12&1][a12+b12])&0x0f;
	d12=a12;a12=b12;b12=n[point];
	
	double random=(((((((((((b12/16.0+b11)/16.0+b10)/16.0+b9)/16.0
			      +b8)/16.0+b7)/16.0+b6)/16.0+b5)/16.0+b4)/16.0
			 +b3)/16.0+b2)/16.0+b1)/16.0;
	return random;
#else
	unsigned long L1=0xB+0xDEECE66DL*(unsigned long)(x2);
	unsigned long L2=0x5*(unsigned long)(x2)+0xDEECE66DL*(unsigned long)(1);
	x2=(unsigned int)(L1);
	L1>>=32;
	x1=(unsigned int)((L2+L1)&0xffffL);
	double random=(x2*((1./65536.)*(1./65536.))+x1)*(1./65536.);
	random=random>=1?0:random;
	return random;
#endif
    }
};


} // end namespace rtrt

#endif
