/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Grid/BoundaryConditions/UnstructuredRectangleBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace Uintah;

UnstructuredRectangleBCData::UnstructuredRectangleBCData() : UnstructuredBCGeomBase()
{
  
}

UnstructuredRectangleBCData::UnstructuredRectangleBCData(Point& low, Point& up) 
: UnstructuredBCGeomBase(),
  d_min(low),
  d_max(up)
{
  d_origin = d_min;
}

UnstructuredRectangleBCData::~UnstructuredRectangleBCData()
{
}


bool UnstructuredRectangleBCData::operator==(const UnstructuredBCGeomBase& rhs) const
{
  const UnstructuredRectangleBCData* p_rhs =
    dynamic_cast<const UnstructuredRectangleBCData*>(&rhs);

  if (p_rhs == nullptr)
    return false;
  else
    return (this->d_min == p_rhs->d_min) && (this->d_max == p_rhs->d_max);
}

UnstructuredRectangleBCData* UnstructuredRectangleBCData::clone()
{
  return scinew UnstructuredRectangleBCData(*this);
}

void UnstructuredRectangleBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}


void UnstructuredRectangleBCData::addBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}

void UnstructuredRectangleBCData::sudoAddBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}

void UnstructuredRectangleBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

bool UnstructuredRectangleBCData::inside(const Point &p) const 
{
  if(d_min.x() == d_max.x()) {
    if (p.y() <= d_max.y() && p.y() >= d_min.y()
        && p.z() <= d_max.z() && p.z() >= d_min.z()) {
      return true;
    }
    else 
      return false;
  }    
  
  else if(d_min.y() == d_max.y()) {
    if (p.x() <= d_max.x() && p.x() >= d_min.x()
        && p.z() <= d_max.z() && p.z() >= d_min.z()) {
      return true;
    }
    else 
      return false;
  }    

  else if(d_min.z() == d_max.z()) {
    if (p.y() <= d_max.y() && p.y() >= d_min.y()
        && p.x() <= d_max.x() && p.x() >= d_min.x()) {
      return true;
    }
    else 
      return false;
  }    
  return false;
}

void UnstructuredRectangleBCData::print() 
{
  BC_dbg << "Geometry type = " << typeid(this).name() << std::endl;
  d_bc.print();
}



void UnstructuredRectangleBCData::determineIteratorLimits(UnstructuredPatch::FaceType face, 
                                              const UnstructuredPatch* patch, 
                                              std::vector<Point>& test_pts)
{
  UnstructuredBCGeomBase::determineIteratorLimits(face,patch,test_pts);
}


