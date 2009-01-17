#include <iostream>
#include <stdlib.h>

extern "C"{
 void bombed_(char *mes);
 void logmes_(char *mes);
 void faterr_(char *mes1, char *mes2);
}

using namespace std;

void bombed_(char *mes)
{
  cerr <<  "Code bombed with the following message:" << endl;
  cerr <<  mes << endl;
  exit(1);
  return;
}

void logmes_(char *mes)
{
  cerr <<  mes << endl;

  return;
}

void faterr_(char *mes1, char *mes2)
{
  cerr << "Fatal error detected by " << mes1 << ":" << endl;
  cerr <<  mes2 << endl;
  exit(1);

  return;
}
