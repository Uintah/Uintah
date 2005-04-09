
#include <stdlib.h>
#include <stdio.h>

int al[] = {
    4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536,
};

main()
{
    unsigned long int tot=0;
    unsigned long int bytes=0;
    for(int i=0;i<2000;i++){
	int n=rand()%1000+1;
	void* p[1000];
	int j;
	for(j=0;j<n;j++){
	    int s=rand()%(65536-8);
	    int a=rand()%(sizeof(al)/sizeof(int));
	    p[j]=memalign(al[a], s);
	    if((unsigned long)p[j] % al[a]){
		fprintf(stderr, "misaligned: %p!\n", p[j]);
	    }
	    bytes+=s;
	}
	for(j=0;j<n;j++){
	    free(p[j]);
	}
	tot+=n;
    }
    fprintf(stderr, "%u allocations (%u bytes)\n", tot, bytes);
    return 0;
}
