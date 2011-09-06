/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Grid/BoundaryConditions/RectangleBCData.h>
#include <Core/Geometry/Point.h>
#include <Core/Grid/Box.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>
#include <iostream>
using namespace SCIRun;
using namespace Uintah;
using std::cout;
using std::endl;

// export SCI_DEBUG="BC_dbg:+"
static DebugStream BC_dbg("BC_dbg",false);

RectangleBCData::RectangleBCData() : BCGeomBase()
{
  
}

RectangleBCData::RectangleBCData(Point& low, Point& up) : BCGeomBase()
{
//  cout << "low = " << low << " up = " << up << endl;
  Point n_low(0.,0.,0.), n_up(0.,0.,0.);

  if(low.x() == up.x()) {
    n_low = Point(low.x()-1.e-2,low.y(),low.z());
    n_up = Point(up.x()+1.e-2,up.y(),up.z());
  }    
  if(low.y() == up.y()) {
    n_low = Point(low.x(),low.y()-1.e-2,low.z());
    n_up = Point(up.x(),up.y()+1.e-2,up.z());
  }    
  if(low.z() == up.z()) {
    n_low = Point(low.x(),low.y(),low.z()-1.e-2);
    n_up = Point(up.x(),up.y(),up.z()+1.e-2);
  }    
  d_min = n_low;
  d_max = n_up;

//  cout << "d_min = " << d_min << " d_max = " << d_max << endl;
}

RectangleBCData::~RectangleBCData()
{
}


bool RectangleBCData::operator==(const BCGeomBase& rhs) const
{
  const RectangleBCData* p_rhs =
    dynamic_cast<const RectangleBCData*>(&rhs);

  if (p_rhs == NULL)
    return false;
  else
    return (this->d_min == p_rhs->d_min) && (this->d_max == p_rhs->d_max);
}

RectangleBCData* RectangleBCData::clone()
{
  return scinew RectangleBCData(*this);
}

void RectangleBCData::addBCData(BCData& bc)
{
  d_bc = bc;
}


void RectangleBCData::addBC(BoundCondBase* bc)
{
  d_bc.setBCValues(bc);
}

void RectangleBCData::getBCData(BCData& bc) const
{
  bc = d_bc;
}

bool RectangleBCData::inside(const Point &p) const 
{
  if (p == Max(p,d_min) && p == Min(p,d_max) )
    return true;
  else 
    return false;
}

void RectangleBCData::print() 
{
  BC_dbg << "Geometry type = " << typeid(this).name() << endl;
  d_bc.print();
}



void RectangleBCData::determineIteratorLimits(Patch::FaceType face, 
                                              const Patch* patch, 
                                              vector<Point>& test_pts)
{
#if 0
  cout << "RectangleBC determineIteratorLimits()" << endl;
#endif

  BCGeomBase::determineIteratorLimits(face,patch,test_pts);

}


