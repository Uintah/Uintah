#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Grid.h>
#include "NCTensorField.h"
using std::vector;


namespace SCICore{
  namespace Datatypes{


NCTensorField::NCTensorField()
  :TensorField()
{
}


NCTensorField::NCTensorField(const NCTensorField& copy)
  //  : UintahScalarField( copy )
  :TensorField( copy ), grid(copy.grid), level(copy.level),
    _varname(copy._varname), _matIndex(copy._matIndex)
{
  for(int i = 0; i < copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
}


NCTensorField::NCTensorField(GridP grid, LevelP level,
				string var, int mat,
				const vector< NCVariable<Matrix3> >& vars)
  //  : UintahScalarField( grid, level, var, mat )
  : TensorField(), grid(grid), level(level),
    _varname(var), _matIndex(mat)
{
  for(int i = 0; i < vars.size(); i++){
    _vars.push_back( vars[i]);
  }
}



void NCTensorField::AddVar( const NCVariable<Matrix3>& v)
{
  _vars.push_back( v );
}


TensorField*  NCTensorField::clone()
{
  return scinew NCTensorField(*this);
} 


void NCTensorField::compute_bounds()
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


void NCTensorField::get_boundary_lines(Array1<Point>& lines)
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
  
} // Datatypes
} // SCICore
