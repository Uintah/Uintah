#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Grid.h>
#include "CCVectorField.h"
#include <iostream>
using std::cerr;
using std::endl;
using std::vector;


namespace SCICore{
  namespace Datatypes{



CCVectorField::CCVectorField()
  :VectorFieldRG()
{
}


CCVectorField::CCVectorField(const CCVectorField& copy)
  //  : UintahScalarField( copy )
  :VectorFieldRG( copy ), grid(copy.grid), level(copy.level),
    _varname(copy._varname), _matIndex(copy._matIndex)
{
  for(int i = 0; i < copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
}


CCVectorField::CCVectorField(GridP grid, LevelP level,
				string var, int mat,
				const vector< CCVariable<Vector> >& vars)
  //  : UintahScalarField( grid, level, var, mat )
  : VectorFieldRG(), grid(grid), level(level),
    _varname(var), _matIndex(mat)
{
  for(int i = 0; i < vars.size(); i++){
    _vars.push_back( vars[i]);
  }
}



void CCVectorField::AddVar( const CCVariable<Vector>& v)
{
  _vars.push_back( v );
}


VectorField*  CCVectorField::clone()
{
  return scinew CCVectorField(*this);
} 


void CCVectorField::compute_bounds()
{
  if(have_bounds || _vars.size() == 0)
    return;
 
  Point min(1e30,1e30,1e30);
  Point max(-1e30,-1e30,-1e30);
 
  for(Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); r++){
    min = SCICore::Geometry::Min( min, (*r)->getBox().lower());
    max = SCICore::Geometry::Max( max, (*r)->getBox().upper());
  }

  bmin = min;
  bmax = max;
  have_bounds = 1;
}


void CCVectorField::get_boundary_lines(Array1<Point>& lines)
{
    Point min, max;
    get_bounds(min, max);
    int i;
    for(i=0;i<4;i++){
	double x=(i&1)?min.x():max.x();
	double y=(i&2)?min.y():max.y();
	lines.add(Point(x, y, min.z()));
	lines.add(Point(x, y, max.z()));
    }
    for(i=0;i<4;i++){
	double y=(i&1)?min.y():max.y();
	double z=(i&2)?min.z():max.z();
	lines.add(Point(min.x(), y, z));
	lines.add(Point(max.x(), y, z));
    }
    for(i=0;i<4;i++){
	double x=(i&1)?min.x():max.x();
	double z=(i&2)?min.z():max.z();
	lines.add(Point(x, min.y(), z));
	lines.add(Point(x, max.y(), z));
    }
}



int CCVectorField::interpolate(const Point& p, Vector& value, int&,
                                   int) {
    return interpolate(p, value);
}


int CCVectorField::interpolate(const Point& p, Vector& value)
{
  using SCICore::Math::Interpolate;

  Uintah::Box b;
  int i;
  Level::const_patchIterator r;
  for(i = 0, r = level->patchesBegin();
      r != level->patchesEnd(); r++, i++){
    b = (*r)->getBox();
    if(b.contains(p)){
      break;
    }
  }

  if (i >= _vars.size())
    return 0;
  
  IntVector index;
  if( !(*r)->findCell( p, index))
    return 0;
  int ix = index.x();
  int iy = index.y();
  int iz = index.z();
  value = _vars[i][IntVector(ix, iy, iz)];
  return 1;
}

} // Datatypes
} // SCICore

