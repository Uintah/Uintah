#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Datatypes/CCTensorField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <sci_values.h>
#include <iostream>

using std::vector;
using std::cerr;
using std::endl;

namespace Uintah {

CCTensorField::CCTensorField()
  :TensorField()
{
}


CCTensorField::CCTensorField(const CCTensorField& copy)
  //  : UintahScalarField( copy )
  :TensorField( copy ), _grid(copy._grid), _level(copy._level),
    _varname(copy._varname), _matIndex(copy._matIndex),
   high(copy.high), low(copy.low), nx(copy.nx),
   ny(copy.ny), nz(copy.nz)
{
  for(int i = 0; i < (int)copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
}


CCTensorField::CCTensorField(GridP grid, LevelP level,
				string var, int mat,
				const vector< CCVariable<Matrix3> >& vars)
  //  : UintahScalarField( grid, level, var, mat )
  : TensorField(), _grid(grid), _level(level),
    _varname(var), _matIndex(mat),
   high(-INT_MAX,-INT_MAX,-INT_MAX),
   low(INT_MAX,INT_MAX,INT_MAX)
{
  for(int i = 0; i < (int)vars.size(); i++){
    _vars.push_back( vars[i]);
  }
  computeHighLowIndices();
  //cerr<<"low index = "<<low<<", high index = "<<high<<endl;
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
}

void CCTensorField::computeHighLowIndices()
{
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++){
    low = Min( low, (*r)->getNodeLowIndex());
    high = Max( high, (*r)->getNodeHighIndex());
  }
}

Matrix3
CCTensorField::grid(int i, int j, int k)
{
  IntVector id(i,j,k);
  id = low + id;
  int ii = 0;
  Matrix3 m;
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++, ii++){
      if( (*r)->containsNode( id ) )
	return _vars[ii][id];
  }
  return m;
}


void CCTensorField::AddVar( const CCVariable<Matrix3>& v)
{
  _vars.push_back( v );
}


TensorField*  CCTensorField::clone()
{
  return scinew CCTensorField(*this);
} 


void CCTensorField::compute_bounds()
{
  if(have_bounds || _vars.size() == 0)
    return;
 
  Point min(1e30,1e30,1e30);
  Point max(-1e30,-1e30,-1e30);
 
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++){
    min = Min( min, (*r)->getBox().lower());
    max = Max( max, (*r)->getBox().upper());
  }

  bmin = min;
  bmax = max;
  have_bounds = 1;
}


void CCTensorField::get_boundary_lines(Array1<Point>& lines)
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

} // End namespace Uintah
