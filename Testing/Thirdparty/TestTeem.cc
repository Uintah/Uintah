#include <teem/nrrd.h>
#include <stdio.h>

int
main(int argc, char *argv[])
{
  fprintf(stdout, "Creating Nrrd\n");
  Nrrd *nrrd=nrrdNew();
  fprintf(stdout, "Done\n");
  return 0; 
}
