#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Core/Geometry/Point.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>

using namespace SCIRun;
using namespace Uintah;
using namespace std;

BCGeomBase::BCGeomBase() 
{
}

BCGeomBase::~BCGeomBase()
{
}

void BCGeomBase::setBoundaryIterator(vector<IntVector>& b)
{
  boundary = b;
}

void BCGeomBase::setNBoundaryIterator(vector<IntVector>& b)
{
  nboundary = b;
}

void BCGeomBase::setInteriorIterator(vector<IntVector>& i)
{
  interior = i;
}

void BCGeomBase::getBoundaryIterator(vector<IntVector>& b) const
{
  b = boundary;
}

void BCGeomBase::getNBoundaryIterator(vector<IntVector>& b) const
{
  b = nboundary;
}

void BCGeomBase::getInteriorIterator(vector<IntVector>& i) const
{
  i = interior;
}
