#include <stdio.h>
#include <unistd.h>
#include <Packages/Yarden/Core/Datatypes/Clock.h>
#include <values.h>

namespace Yarden {
unsigned int cycleval;

#ifdef __sgi

volatile iotimer_t counter_value, *iotimer_addr;


void
init_clock()
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
  
//   printf("num pico per tick: %d\n",cycleval);
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
  printf("%s %lf %lf\n", txt,diff*(cycleval*1.0)*1E-12,
	 1.0/( diff*(cycleval*1.0)*1E-12));
}

#else

void init_clock() 
{
  cycleval = 1000;
}

#endif
} // End namespace Yarden

