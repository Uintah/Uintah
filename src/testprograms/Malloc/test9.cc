
#include <stdlib.h>
#include <stdio.h>

int
main(char **, int )
{
    unsigned long int tot=0;
    unsigned long int bytes=0;
    for(int i=0;i<1000;i++){
	int n=rand()%1000+1;
	void* p[1000];
	int j;
	for(j=0;j<n;j++){
	    int s=rand()%504;
	    p[j]=malloc(s);
	    bytes+=s;
	}
	for(j=0;j<n;j++){
	    free(p[j]);
	}
	tot+=n;
    }
    fprintf(stderr, "%lu allocations (%lu bytes)\n", tot, bytes);
    return 0;
}
