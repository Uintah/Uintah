#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>

int getNumNonEmptyLines(char *fname) {
  // read through the file -- when you see a non-white-space set a flag to one.
  // when you get to the end of the line (or EOF), see if the flag has
  // been set.  if it has, increment the count and reset the flag to zero.

  FILE *fin = fopen(fname, "rt");
  if (!fin) {
    std::cerr << "Error -- could not open file "<<fname<<"\n";
    exit(0);
  }
  int count=0;
  int haveNonWhiteSpace=0;
  int c;
  while ((c=fgetc(fin)) != EOF) {
    if (!isspace(c)) haveNonWhiteSpace=1;
    else if (c=='\n' && haveNonWhiteSpace) {
      count++;
      haveNonWhiteSpace=0;
    }
  }
  if (haveNonWhiteSpace) count++;
  std::cerr << "number of nonempty lines was: "<<count<<"\n";
  return count;
}

