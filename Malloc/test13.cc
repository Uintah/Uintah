
#include <stdlib.h>
#include <stdio.h>
#include <bstring.h>

main()
{
    unsigned long int tot=0;
    unsigned long int bytes=0;
    for(int i=0;i<50;i++){
	int n=rand()%10+1;
	void* p[10];
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
