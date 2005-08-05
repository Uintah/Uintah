#include "Vector.h"

static void
instantiate(void)
{
  {
    Vector<Counter> a;
    int b;
    Vector<int> c;
    c = a + b;
    c = a - b;
  }

  if (0) {
    instantiate();
  }
}
