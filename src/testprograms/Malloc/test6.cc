
#include <stdlib.h>
#include <stdio.h>

main()
{
    fprintf(stderr, "This should fail - wrote to object after free\n\n\n");
    void* p=malloc(8);
    free(p);
    int* i=(int*)p;
    *i=0;
    p=malloc(8);
    return 0;
}
