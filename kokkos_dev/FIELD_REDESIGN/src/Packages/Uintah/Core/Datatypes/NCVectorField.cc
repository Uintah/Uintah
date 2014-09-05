#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/MinMax.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Handle.h>
#include <Uintah/Grid/Grid.h>
#include "NCVectorField.h"
using std::vector;


namespace SCICore{
  namespace Datatypes{


NCVectorField::NCVectorField()
  :VectorFieldRG( )
{
}


NCVectorField::NCVectorField(const NCVectorField& copy)
  //  : UintahScalarField( copy )
  :VectorFieldRG( copy ), grid(copy.grid), _level(copy._level),
    _varname(copy._varname), _matIndex(copy._matIndex)
{
  for(int i = 0; i < (int)copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
}


NCVectorField::NCVectorField(GridP grid, LevelP level,
				string var, int mat,
				const vector< NCVariable<Vector> >& vars)
  //  : UintahScalarField( grid, level, var, mat )
  : VectorFieldRG( ), grid(grid), _level(level),
    _varname(var), _matIndex(mat)
{
  for(int i = 0; i < (int)vars.size(); i++){
    _vars.push_back( vars[i]);
  }
}



void NCVectorField::AddVar( const NCVariable<Vector>& v)
{
  _vars.push_back( v );
}


VectorField*  NCVectorField::clone()
{
  return scinew NCVectorField(*this);
} 


void NCVectorField::compute_bounds()
{
  if(have_bounds || _vars.size() == 0)
    return;
 
  Point min(1e30,1e30,1e30);
  Point max(-1e30,-1e30,-1e30);
 
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++){
    min = SCICore::Geometry::Min( min, (*r)->getBox().lower());
    max = SCICore::Geometry::Max( max, (*r)->getBox().upper());
  }

  bmin = min;
  bmax = max;
  have_bounds = 1;
}


void NCVectorField::get_boundary_lines(Array1<Point>& lines)
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

int NCVectorField::interpolate(const Point& p, Vector& value, int&, int)
{
    return interpolate(p, value);
}


int NCVectorField::interpolate(const Point& p, Vector& value)
{
  using SCICore::Math::Interpolate;
  Level::const_patchIterator r;
  int i;
  for(i = 0, r = _level->patchesBegin();
      r != _level->patchesEnd(); r++, i++){
    
    if (i >= (int)_vars.size())
      return 0;

    IntVector ni[8];
    double S[8];
    if( (*r)->findCell( p, *ni ) ) {
      (*r)->findCellAndWeights(p, ni, S);
      value=Vector(0,0,0);
      for(int k = 0; k < 8; k++){
	if((*r)->containsNode(ni[k])){
	  value += _vars[i][ni[k]]*S[k];
	} else {
	  Level::const_patchIterator r1;
	  int j;
	  for(j = 0, r1 = _level->patchesBegin();
	      r1 != _level->patchesEnd(); r1++, j++)
	    if( (*r1)->containsNode(ni[k])){
	      value += _vars[j][ni[k]]*S[k];
	      break;
	    }
	}
      }
      return 1;
    }
  }
  return 0;
}

  
} // Datatypes
} // SCICore
