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

#include <CCA/Components/MPM/PhysicalBC/CrackBC.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <iostream>

using namespace Uintah;
using namespace std;

CrackBC::CrackBC(ProblemSpecP& ps)
{
  Point c1,c2,c3,c4;
  ps->require("corner1",c1);
  ps->require("corner2",c2);
  ps->require("corner3",c3);
  ps->require("corner4",c4);
  
  cout<<"corner1:"<<c1<<endl;
  cout<<"corner2:"<<c2<<endl;
  cout<<"corner3:"<<c3<<endl;
  cout<<"corner4:"<<c4<<endl;
  
  d_origin = c1;
  
  d_e1 = c2-c1;
  d_e1.normalize();
  
  d_e3 = Cross(c1.asVector(),c2.asVector()) + 
         Cross(c2.asVector(),c3.asVector()) + 
         Cross(c3.asVector(),c4.asVector()) + 
         Cross(c4.asVector(),c1.asVector());
  d_e3.normalize();
  
  d_e2 = Cross(d_e3,d_e1);
  
  Vector d;

  d_x1 = 0; d_y1 = 0;

  d = c2 - d_origin;
  d_x2 = Dot(d,d_e1);
  d_y2 = Dot(d,d_e2);
  
  d = c3 - d_origin;
  d_x3 = Dot(d,d_e1);
  d_y3 = Dot(d,d_e2);

  d = c4 - d_origin;
  d_x4 = Dot(d,d_e1);
  d_y4 = Dot(d,d_e2);
}

void CrackBC::outputProblemSpec(ProblemSpecP& ps)
{

}

double CrackBC::x1() const
{
  return d_x1;
}

double CrackBC::y1() const
{
  return d_y1;
}

double CrackBC::x2() const
{
  return d_x2;
}

double CrackBC::y2() const
{
  return d_y2;
}

double CrackBC::x3() const
{
  return d_x3;
}

double CrackBC::y3() const
{
  return d_y3;
}

double CrackBC::x4() const
{
  return d_x4;
}

double CrackBC::y4() const
{
  return d_y4;
}

const Vector& CrackBC::e1() const
{
  return d_e1;
}

const Vector& CrackBC::e2() const
{
  return d_e2;
}

const Vector& CrackBC::e3() const
{
  return d_e3;
}

const Point& CrackBC::origin() const
{
  return d_origin;
}

std::string CrackBC::getType() const
{
  return "Crack";
}
