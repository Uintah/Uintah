/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */



#include <Core/Grid/Region.h>

#include <iostream>
using namespace std;
using namespace Uintah;

int main()
{
  vector<Region> r1, r2, difference;

  r1.push_back(Region(IntVector(0,0,0),IntVector(4,4,4)));
  r1.push_back(Region(IntVector(4,4,4),IntVector(8,8,8)));
  r2.push_back(Region(IntVector(0,0,0),IntVector(2,2,2)));
  r2.push_back(Region(IntVector(2,2,2),IntVector(4,4,4)));

  difference=Region::difference(r1,r2);

  for(vector<Region>::iterator iter=difference.begin();iter!=difference.end();iter++)
  {
    cout << *iter << endl;
  }
}
