
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#ifdef __sun
  #include <string.h>
  #define bcopy(src,dest,n) memcpy(dest,src,n)
#else
  #ifndef __linux
    #include <bstring.h>
  #endif
#endif

main()
{
    for(int i=0;i<50;i++){
	int n=rand()%1000+1;
	void* p=malloc(10);
	for(int j=0;j<n;j++){
	    int s=rand()%(65536-8);
	    p=realloc(p, s);
	    bzero(p, s);
	}
	free(p);
    }
    return 0;
}
