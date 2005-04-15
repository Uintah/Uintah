
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
    unsigned long int tot=0;
    unsigned long int bytes=0;
    for(int i=0;i<100;i++){
	int n=rand()%40+1;
	void* p[40];
	int j;
	for(j=0;j<n;j++){
	    int s=(rand()|(rand()<<15))%(2*1024*1024);
	    p[j]=malloc(s);
	    bytes+=s;
	    bzero(p[j], s);
	}
	for(j=0;j<n;j++){
	    free(p[j]);
	}
	tot+=n;
    }
    fprintf(stderr, "%u allocations (%u bytes)\n", tot, bytes);
    return 0;
}
