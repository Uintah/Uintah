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
  :VectorFieldRG( copy ), grid(copy.grid), level(copy.level),
    _varname(copy._varname), _matIndex(copy._matIndex)
{
  for(int i = 0; i < copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
}


NCVectorField::NCVectorField(GridP grid, LevelP level,
				string var, int mat,
				const vector< NCVariable<Vector> >& vars)
  //  : UintahScalarField( grid, level, var, mat )
  : VectorFieldRG( ), grid(grid), level(level),
    _varname(var), _matIndex(mat)
{
  for(int i = 0; i < vars.size(); i++){
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
 
  for(Level::const_patchIterator r = level->patchesBegin();
      r != level->patchesEnd(); r++){
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
  int ix1 = ix+1;
  int iy1 = iy+1;
  int iz1 = iz+1;
  IntVector upper = (*r)->getNodeHighIndex();
  IntVector lower = (*r)->getNodeLowIndex();
  Vector diag = b.upper() - b.lower();
  Vector pn = p - b.lower();
  double x = pn.x()*(upper.x() - lower.x()-1)/diag.x();
  double y = pn.y()*(upper.y() - lower.y()-1)/diag.y();
  double z = pn.z()*(upper.z() - lower.z()-1)/diag.z();


    if (ix1>=upper.x()) { ix1=ix; }
    if (iy1>=upper.y()) { iy1=iy; }
    if (iz1>=upper.z()) { iz1=iz; }
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    typedef IntVector iv;
    Vector x00= SCICore::Geometry::Interpolate(_vars[i][iv(ix, iy, iz)], 
			   _vars[i][iv(ix1, iy, iz)], fx);
    Vector x01=SCICore::Geometry::Interpolate(_vars[i][iv(ix, iy, iz1)],
			   _vars[i][iv(ix1, iy, iz1)], fx);
    Vector x10=SCICore::Geometry::Interpolate(_vars[i][iv(ix, iy1, iz)],
			   _vars[i][iv(ix1, iy1, iz)], fx);
    Vector x11=SCICore::Geometry::Interpolate(_vars[i][iv(ix, iy1, iz1)],
			   _vars[i][iv(ix1, iy1, iz1)], fx);
    Vector y0=SCICore::Geometry::Interpolate(x00, x10, fy);
    Vector y1=SCICore::Geometry::Interpolate(x01, x11, fy);
    value=SCICore::Geometry::Interpolate(y0, y1, fz);
    return 1;
}


  
} // Datatypes
} // SCICore
