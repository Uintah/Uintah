
#include <stdlib.h>
#include <stdio.h>


int
main(char **, int )
{
    fprintf(stderr, "This should fail - freeing a pointer twice...\n\n\n");
    void* p=malloc(20);
    free(p);
    free(p);
    return 0;
}
