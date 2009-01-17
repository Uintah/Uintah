#include <iostream>
#include <stdlib.h>

extern "C"{
 int bombed_(char *mes);
 int logmes_(char *mes);
 int tokens_(char *mes);
 int faterr_(char *mes1, char *mes2);
}

using namespace std;

int bombed_(char *mes)
{
  cerr <<  mes << endl;
  exit(1);
  return 0;
}

int logmes_(char *mes)
{
  cerr <<  mes << endl;

  return 1;
}

int tokens_(char *mes)
{
  cerr <<  mes << endl;

  return 1;
}

int faterr_(char *mes1, char *mes2)
{
  cerr <<  mes1 << " " << mes2 << endl;
  exit(1);

  return 0;
}
