#include "nrrd.h"

/*
******** nrrdStr2Type()
**
** takes a given string and returns the integral type
*/
int
nrrdStr2Type(char *str) {

  /* >>>> obviously, the next two arrays      <<<<
     >>>> have to be in sync with each other, <<<<
     >>>> and the second has be in sync with  <<<<
     >>>> the nrrdType enum in nrrd.h         <<<< */

#define NUMVARIANTS 22

  char typStr[NUMVARIANTS][NRRD_SMALL_STRLEN]  = {
    "char", "signed char",
    "unsigned char",
    "short", "short int", "signed short", "signed short int",
    "unsigned short", "unsigned short int",
    "int", "signed int",
    "unsigned int",
    "long long", "long long int", "signed long long", "signed long long int",
    "unsigned long long", "unsigned long long int",
    "float",
    "double",
    "long double",
    "block"};
  int numStr[NUMVARIANTS] = {
    1, 1,
    2,
    3, 3, 3, 3,
    4, 4,
    5, 5,
    6,
    7, 7, 7, 7,
    8, 8,
    9,
    10,
    11,
    12};

  int i;

  for (i=0; i<=NUMVARIANTS-1; i++) {
    if (!(strcmp(typStr[i], str))) {
      return(numStr[i]);
    }
  }
  return(nrrdTypeUnknown);
}


