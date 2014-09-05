#include "ppm.h"
#include <stdio.h>

void write_PPM( const char *filename, unsigned char * buffer,
		       int width, int height)
{
  int i;
  FILE *out;

  out = fopen(filename, "w");
  if (!out) {
    fprintf(stderr, "Can't open output file %s\n", filename);
    return;
  }
  fprintf(out, "P6\n");   // color rawbits format
  fprintf(out, "%d %d\n%d\n", width, height, 255);  // width, height, and depth

  for (i = 0; i < width * height*3; i++)
    fprintf(out, "%c", buffer[i]);
    
  fprintf(out, "\n");
  fclose(out);
}

