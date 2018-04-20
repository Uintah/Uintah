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

#include <Core/Grid/BoundaryConditions/RectangulusBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace Uintah;

RectangulusBCData::RectangulusBCData() : BCGeomBase()
{
  
}

RectangulusBCData::RectangulusBCData(Point& low_in, Point& up_in, Point& low_out, Point& up_out)
: BCGeomBase(),
  d_min_in(low_in),
  d_max_in(up_in),
  d_min_out(low_out),
  d_max_out(up_out)
{
}

RectangulusBCData::~RectangulusBCData()
{
}


bool RectangulusBCData::operator==(const BCGeomBase& rhs) const
{
  const RectangulusBCData* p_rhs = 
    dynamic_cast<const RectangulusBCData*>(&rhs);

  if (p_rhs == nullptr)
    return false;
  else
    return (this->d_min_out == p_rhs->d_min_out) &&
           (this->d_max_out == p_rhs->d_max_out) &&
           (this->d_min_in == p_rhs->d_min_in) &&
           (this->d_max_in == p_rhs->d_max_in);
}

RectangulusBCData* RectangulusBCData::clone()
{
  return scinew RectangulusBCData(*this);
}

void RectangulusBCData::addBCData(BCData& bc) 
{
  d_bc = bc;
}


void RectangulusBCData::addBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void RectangulusBCData::sudoAddBC(BoundCondBase* bc) 
{
  d_bc.setBCValues(bc);
}

void RectangulusBCData::getBCData(BCData& bc) const 
{
  bc = d_bc;
}

bool RectangulusBCData::inside(const Point &p) const 
{
  if(d_min_out.x() == d_max_out.x() && d_min_in.x() == d_max_in.x()) { // x face inlet
    if (p.y() <= d_max_out.y() && p.y() >= d_min_out.y() && p.z() <= d_max_out.z() && p.z() >= d_min_out.z()) {
      if (p.y() <= d_max_in.y() && p.y() >= d_min_in.y() && p.z() <= d_max_in.z() && p.z() >= d_min_in.z()) {
        return false; // if you are inside of the outer box and the inner box then you are not in the rectangulus.
      } else {
        return true; // if you are in the outer box but not in the inner box then are in the rectangulus.
      }
    } else { 
      return false; // if you are not inside of the outer box then you are not in the rectangulus.   
    }  
  }
  else if(d_min_out.y() == d_max_out.y() && d_min_in.y() == d_max_in.y()) { // y face inlet
    if (p.x() <= d_max_out.x() && p.x() >= d_min_out.x() && p.z() <= d_max_out.z() && p.z() >= d_min_out.z()) {
      if (p.x() <= d_max_in.x() && p.x() >= d_min_in.x() && p.z() <= d_max_in.z() && p.z() >= d_min_in.z()) {
        return false; // if you are inside of the outer box and the inner box then you are not in the rectangulus.
      } else {
        return true; // if you are in the outer box but not in the inner box then are in the rectangulus.
      }
    } else { 
      return false; // if you are not inside of the outer box then you are not in the rectangulus.   
    }  
  }    
  else if(d_min_out.z() == d_max_out.z() && d_min_in.z() == d_max_in.z()) { // z face inlet
    if (p.x() <= d_max_out.x() && p.x() >= d_min_out.x() && p.y() <= d_max_out.y() && p.y() >= d_min_out.y()) {
      if (p.x() <= d_max_in.x() && p.x() >= d_min_in.x() && p.y() <= d_max_in.y() && p.y() >= d_min_in.y()) {
        return false; // if you are inside of the outer box and the inner box then you are not in the rectangulus.
      } else {
        return true; // if you are in the outer box but not in the inner box then are in the rectangulus.
      }
    } else { 
      return false; // if you are not inside of the outer box then you are not in the rectangulus.   
    }  
  }    
  return false;
}

void RectangulusBCData::print()
{
  BC_dbg << "Geometry type = " << typeid(this).name() << std::endl;
  d_bc.print();
}



void RectangulusBCData::determineIteratorLimits(Patch::FaceType face, 
                                              const Patch* patch, 
                                              std::vector<Point>& test_pts)
{
  BCGeomBase::determineIteratorLimits(face,patch,test_pts);
}


