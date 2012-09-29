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

#include <Core/Grid/BoundaryConditions/CircleBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <iostream>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

// export SCI_DEBUG="BC_dbg:+"
static DebugStream BC_dbg("BC_dbg",false);

CircleBCData::CircleBCData() : BCGeomBase()
{
  
}


CircleBCData::CircleBCData(Point& p, double radius)
  : BCGeomBase(), d_radius(radius), d_origin(p)
{
}

CircleBCData::~CircleBCData()
{
}

bool CircleBCData::operator==(const BCGeomBase& rhs) const
{

  const CircleBCData* p_rhs =
    dynamic_cast<const CircleBCData*>(&rhs);

  if (p_rhs == NULL)
    return false;
  else
    return (this->d_radius == p_rhs->d_radius) && 
      (this->d_origin == p_rhs->d_origin);

}

CircleBCData* CircleBCData::clone()
{
  return scinew CircleBCData(*this);
}

void CircleBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void CircleBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void CircleBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool CircleBCData::inside(const Point &p) const 
{
  Vector diff = p - d_origin;
  if (diff.length() > d_radius)
    return false;
  else
    return true;
}

void CircleBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}


void CircleBCData::determineIteratorLimits(Patch::FaceType face, 
                                           const Patch* patch, 
                                           vector<Point>& test_pts)
{
#if 0
  cout << "Circle determineIteratorLimits() " << patch->getFaceName(face)<< endl;
#endif

  BCGeomBase::determineIteratorLimits(face,patch,test_pts);

}

