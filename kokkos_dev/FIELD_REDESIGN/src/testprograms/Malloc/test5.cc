
#include <stdlib.h>
#include <stdio.h>

main()
{
    fprintf(stderr, "This should fail - wrote after object\n\n\n");
    void* p=malloc(8);
    int* i=(int*)p;
    i+=2;
    *i=0;
    free(p);
    return 0;
}
