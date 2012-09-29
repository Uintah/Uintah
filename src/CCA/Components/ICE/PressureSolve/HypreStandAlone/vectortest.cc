/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
