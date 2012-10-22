/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#include <cstdlib>

class Point
{
public:
  Point()
    : d_id(-999999) {}
  
  Point(int id, int x, int y, int z)
    : d_id(id)
  { d_vals[0] = x; d_vals[1] = y; d_vals[2] = z; }

  /*
  Point(const Point& p)
    : d_id(p.d_id)
  {
    d_vals[0] = p.d_vals[0]; d_vals[1] = p.d_vals[1]; d_vals[2] = p.d_vals[2];
    }*/
    
  Point& operator=(const Point& p)
  {
    d_id = p.d_id;
    d_vals[0] = p.d_vals[0]; d_vals[1] = p.d_vals[1]; d_vals[2] = p.d_vals[2];
    return *this;
  }

  int distanceSquared(const Point& p2) const
  { return (d_vals[0] - p2.d_vals[0]) * (d_vals[0] - p2.d_vals[0]) +
      (d_vals[1] - p2.d_vals[1]) * (d_vals[1] - p2.d_vals[1]) +
      (d_vals[2] - p2.d_vals[2]) * (d_vals[2] - p2.d_vals[2]); }
  
  int distanceL1(const Point& p2) const
  { return (int)std::abs(d_vals[0] - p2.d_vals[0])
      + (int)std::abs(d_vals[1] - p2.d_vals[1])
      + (int)std::abs(d_vals[2] - p2.d_vals[2]); }
    
  int getId() const
  { return d_id; }
  
  int operator[](int i) const
  { return d_vals[i]; }

  int& operator[](int i)
  { return d_vals[i]; }
private:
  int d_id;
  int d_vals[3];
};
