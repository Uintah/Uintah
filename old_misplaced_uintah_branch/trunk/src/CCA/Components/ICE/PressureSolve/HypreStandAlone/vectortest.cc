#include "Vector.h"
#include "Box.h"

int MYID;
DebugStream dbg("DEBUG",true);
DebugStream dbg0("DEBUG",true);

int main(int argc, char **argv)
{
  {
    Vector<Counter> a;
    int b;
    Vector<int> c;
    c = a + b;
    c = a - b;
  }
  double b = 0.5;
  Vector<double> a(0,5,0,"a",0.3);
  Vector<double> c = b + a;
  dbg0 << a << "\n";
  dbg0 << b << "\n";
  dbg0 << c << "\n";
  dbg0 << a - b << "\n";
  
  Vector<int> lower(0,2,0,"",0);
  Vector<int> upper(0,2,0,"",3);
  lower[0] = 2;
  upper[0] = 2;

  Box box(lower,upper);
  for (Box::iterator iter = box.begin(); iter != box.end(); ++iter) {
    dbg0 << "*iter = " << *iter << "\n";
  }
  return 0;
}
