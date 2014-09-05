
#include <stdlib.h>
#include <stdio.h>

main()
{
    fprintf(stderr, "This should fail - wrote after object\n\n\n");
    void* p=malloc(4);
    int* i=(int*)p;
    i+=1;
    *i=0;
    free(p);
    return 0;
}
