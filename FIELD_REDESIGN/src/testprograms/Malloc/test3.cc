
#include <stdlib.h>
#include <stdio.h>

main()
{
    fprintf(stderr, "This should fail - wrote before object\n\n\n");
    void* p=malloc(20);
    int* i=(int*)p;
    i--;
    *i=0;
    free(p);
    return 0;
}
