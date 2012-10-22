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

#include <Core/Grid/BoundaryConditions/AnnulusBCData.h>
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

AnnulusBCData::AnnulusBCData() : BCGeomBase()
{
  
}

AnnulusBCData::AnnulusBCData(Point& p, double inRadius,double outRadius)
  : BCGeomBase(), d_innerRadius(inRadius), d_outerRadius(outRadius), d_origin(p)
{
}

AnnulusBCData::~AnnulusBCData()
{
}


bool AnnulusBCData::operator==(const BCGeomBase& rhs) const
{
  const AnnulusBCData* p_rhs = 
    dynamic_cast<const AnnulusBCData*>(&rhs);
  
  if (p_rhs == NULL)
    return false;
  else 
    return (this->d_innerRadius == p_rhs->d_innerRadius) && 
      (this->d_outerRadius == p_rhs->d_outerRadius) && 
      (this->d_origin == p_rhs->d_origin) ;
  
}

AnnulusBCData* AnnulusBCData::clone()
{
  return scinew AnnulusBCData(*this);
}

void AnnulusBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void AnnulusBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void AnnulusBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool AnnulusBCData::inside(const Point &p) const 
{
  Vector diff = p - d_origin;

  bool inside_outer = false;
  bool outside_inner = false;

  if (diff.length() > d_outerRadius)
    inside_outer = false;
  else
    inside_outer =  true;

  if (diff.length() > d_innerRadius)
    outside_inner = true;
  else
    outside_inner = false;

  return (inside_outer && outside_inner);
  
}

void AnnulusBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}

void AnnulusBCData::determineIteratorLimits(Patch::FaceType face, 
                                           const Patch* patch, 
                                           vector<Point>& test_pts)
{
#if 0
  cout << "Annulus determineIteratorLimits()" << endl;
#endif

  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
}

