
/*
  Clock.cc


  Copyright (C) 2000 SCI Group, University of Utah
*/

#include <stdio.h>
#include <unistd.h>
#include "Clock.h"
#include <values.h>

namespace Phil {
volatile iotimer_t counter_value, *iotimer_addr;
unsigned int cycleval;


void
init_clock(int n)
{
  __psunsigned_t phys_addr, raddr;
  //  volatile iotimer_t counter_value, *iotimer_addr;
  int fd, poffmask;
  
  
  poffmask = getpagesize() - 1;
  phys_addr = syssgi(SGI_QUERY_CYCLECNTR, &cycleval);
  raddr = phys_addr & ~poffmask;
  fd = open("/dev/mmem", O_RDONLY);
  iotimer_addr = (volatile iotimer_t *)mmap(0, poffmask, PROT_READ,
					    MAP_PRIVATE, fd, (off_t)raddr);
  iotimer_addr = (iotimer_t *)((__psunsigned_t)iotimer_addr +
			       (phys_addr & poffmask));
  counter_value = *(iotimer_addr);

  clocks = new double[n];
  for(int i = 0; i < n; i++)
    clocks[i] = 0.0;
//   printf("num pico per tick: %d\n",cycleval);
}

void
AddTime( iotimer_t s, iotimer_t e, int id ) {
  long long res = e-s;
  iotimer_t diff;
  if (res < 0) { // you wrapped...
    res += MAXLONG; // add max long to this...
    diff = res;
  } else {
    diff = e-s;
  }
  clocks[id] += diff*(cycleval*1.0)*1E-12;
}

void PrintAvgTime(FILE *fp, int id, int n, char *txt) {
  fprintf(fp, "%s %lfs\n",txt, clocks[id] / (double)n);
}

void ClearTime( int id ) {
  clocks[id] = 0.0;
}

void 
fPrintTime(FILE *fp, iotimer_t s, iotimer_t e, char *txt)
{
  long long res = e-s;
  iotimer_t diff;
  
  if (res < 0) { // you wrapped...
    res += MAXLONG; // add max long to this...
    diff = res;
  } else {
    diff = e-s;
  }
  //  fprintf(fp,"%s %lfms\n", txt,diff*(cycleval*1.0)*1E-9);
  fprintf(fp,"%s %lfs\n", txt,diff*(cycleval*1.0)*1E-12);
}

void 
PrintTime(iotimer_t s, iotimer_t e, char *txt)
{
  long long res = e-s;
  iotimer_t diff;
  
  if (res < 0) { // you wrapped...
    res += MAXLONG; // add max long to this...
    diff = res;
  } else {
    diff = e-s;
  }
  //  fprintf(fp,"%s %lfms\n", txt,diff*(cycleval*1.0)*1E-9);
  printf("%s %lfs\n", txt,diff*(cycleval*1.0)*1E-12);
}
} // End namespace Phil


