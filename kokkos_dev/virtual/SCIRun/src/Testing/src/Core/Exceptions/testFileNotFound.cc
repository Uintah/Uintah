#include <Core/Exceptions/FileNotFound.h>

int main()
{
  SCIRun::FileNotFound *ex = new SCIRun::FileNotFound("Error","filename.txt",0);
  delete ex;
  return 0;
}
