#include <stdio.h>
#include <limits.h>

/*
******** BEGIN SNAGGED CODE
**
** This code snippet (with trivial changes) was snagged from 
** http://www.softmall.ibm.com:1100/as400/porting/example16.html
** on Fri Sep. 11
**

This function checks if the passed parameter is not a number (NaN). If
the exponent of an AS/400 double (which is 8 byte) is all 1's and
fraction that is not all 0's then it is infinite. Bits in an AS/400
double are:

           0     sign
           1:11  exponent
           12:63 fraction
                                                                    
Bit 0 is the leftmost bit.

Choose your browser's option to save to local disk and then reload
this document to download this code example. Send the program to your
AS/400 and compile it using the development facilities supplied there.
This program was developed V3R6 system and tested on V3R1, V3R2 and
V3R6 systems.

This small program that is furnished by IBM is a simple example to
provide an illustration. This example has not been thoroughly tested
under all conditions. IBM, therefore, cannot guarantee or imply
reliability, serviceability, or function of this program. All programs
contained herein are provided to you "AS IS". THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE EXPRESSLY
DISCLAIMED.

***/

typedef struct {
  unsigned long lobits;
  unsigned long hibits;
} BITMASK;

typedef union {
  BITMASK bm;
  double  dv;
} OVERLAY;

#define EXPBITS  0x7FF00000
#define ALLZEROS 0x00000000
#define CHKEXP   0x000FFFFF


int 
nrrdIsNand(double fres) {

  OVERLAY value;
  
  value.dv = fres;
  if ( ( (value.bm.lobits & EXPBITS) == EXPBITS )  &&
       ( ( (value.bm.lobits & CHKEXP)  != ALLZEROS ) ||
	 ( value.bm.hibits != ALLZEROS ) ) )
    return(1); /* Yes, it is not a number NaN */
  else
    return(0); /* it is a number */
}

/*
******* END SNAGGED CODE
*/

int
nrrdIsNanf(float val) {
  return(isnand((double)val));
}


float
nrrdNanf(void) {
  static float nanf = 0;
  
  if (0 == nanf) {
    nanf = 1.0;
    nanf /= 0.0;
    nanf /= nanf;
  }
  return(nanf);
}

double
nrrdNand(void) {
  static double nand = 0;
  
  if (0 == nand) {
    nand = 1.0;
    nand /= 0.0;
    nand /= nand;
  }
  return(nand);
}

/*
void
main() {
  double d;
  float f;

  d = 1.0;
  d /= 0.0;
  d /= d;
  printf("%lf: %d\n", d, nrrdIsNand(d));

  d = nrrdNand();
  printf("%lf: %d\n", d, nrrdIsNand(d));

  f = 1.0;
  f /= 0.0;
  f /= f;
  printf("%f: %d\n", f, nrrdIsNanf(f));

  f = nrrdNanf();
  printf("%f: %d\n", f, nrrdIsNanf(f));
}
*/

