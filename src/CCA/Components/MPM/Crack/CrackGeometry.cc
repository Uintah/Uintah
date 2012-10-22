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

#include <CCA/Components/MPM/Crack/CrackGeometry.h>
#include <Core/Geometry/Vector.h>
#include <cmath>

using std::vector;

using namespace Uintah;

CrackGeometry::CrackGeometry()
{
}


CrackGeometry::~CrackGeometry()
{
  // Destructor
  // Do nothing
}

bool CrackGeometry::twoLinesCoincide(Point& p1, Point& p2, Point& p3, 
                                     Point& p4)
{
  // Check for coincidence between line segment p3-p4 and line
  // segment p1 - p2
  double l12 = (p2.asVector() - p1.asVector()).length();
  double l31 = (p3.asVector() - p1.asVector()).length();
  double l32 = (p3.asVector() - p2.asVector()).length();
  double l41 = (p4.asVector() - p1.asVector()).length();
  double l42 = (p4.asVector() - p2.asVector()).length();

  if (fabs(l31+l32-l12)/l12 < 1.e-6 && fabs(l41+l42-l12)/l12 < 1.e-6 && l41>l31)
    return true;
  else
    return false;


}
