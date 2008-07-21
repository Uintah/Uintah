#include <Packages/Uintah/Core/Grid/BoundaryConditions/RectangleBCData.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using namespace SCIRun;
using namespace Uintah;
using std::cout;
using std::endl;

RectangleBCData::RectangleBCData() 
{
  
}

RectangleBCData::RectangleBCData(BCData& bc)
  : d_bc(bc)
{
}

RectangleBCData::RectangleBCData(Point& low, Point& up)
{
  cout << "low = " << low << " up = " << up << endl;
  Point n_low, n_up;

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

  cout << "d_min = " << d_min << " d_max = " << d_max << endl;
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
  cout << "Geometry type = " << typeid(this).name() << endl;
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


