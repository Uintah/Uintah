#include <SCICore/Math/MiscMath.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
#include "NCScalarField.h"
#include <values.h>
using std::vector;


namespace SCICore{
  namespace Datatypes{

using namespace Uintah;
using namespace SCICore::Geometry;

template <class T>
NCScalarField<T>::NCScalarField()
  :ScalarFieldRGBase(),
   high(-MAXINT,-MAXINT,-MAXINT),
   low(MAXINT,MAXINT,MAXINT)
{
}

template <class T>
NCScalarField<T>::NCScalarField(const NCScalarField<T>& copy)
  //  : UintahScalarField( copy )
  :ScalarFieldRGBase( copy ), _grid(copy._grid), _level(copy._level),
    _varname(copy._varname), _matIndex(copy._matIndex), 
   high(-MAXINT,-MAXINT,-MAXINT),
   low(MAXINT,MAXINT,MAXINT)
{
  for(int i = 0; i < copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
  computeHighLowIndices();
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
  
}

template <class T>
NCScalarField<T>::NCScalarField(GridP grid, LevelP level,
				string var, int mat,
				const vector< NCVariable<T> >& vars)
  //  : UintahScalarField( grid, level, var, mat )
  : ScalarFieldRGBase(), _grid(grid), _level(level),
    _varname(var), _matIndex(mat),
   high(-MAXINT,-MAXINT,-MAXINT),
   low(MAXINT,MAXINT,MAXINT)
{
  for(int i = 0; i < vars.size(); i++){
    _vars.push_back( vars[i]);
  }
  computeHighLowIndices();
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
}

template <class T>
void NCScalarField<T>::computeHighLowIndices()
{
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++){
    low = SCICore::Geometry::Min( low, (*r)->getNodeLowIndex());
    high = SCICore::Geometry::Max( high, (*r)->getNodeHighIndex());
  }
}

template <class T>
T NCScalarField<T>::grid(int i, int j, int k)
{
  IntVector id(i,j,k);
  id = low + id;
  int ii = 0;
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++, ii++){
      if( (*r)->containsNode( id ) )
	return _vars[ii][id];
  }
  return 0;
}

template <class T>
double  NCScalarField<T>::get_value(int i, int j, int k)
{
  return double( grid(i,j,k) );
}

template <class T>
void NCScalarField<T>::AddVar( const NCVariable<T>& v, const Patch* p)
{
  _vars.push_back( v );
  low = SCICore::Geometry::Min( low, p->getNodeLowIndex());
  high = SCICore::Geometry::Max( high, p->getNodeHighIndex());
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
}


template<class T>
ScalarField*  NCScalarField<T>::clone()
{
  return scinew NCScalarField<T>(*this);
} 

template<class T>
void NCScalarField<T>::compute_minmax()
{
  T min = 1e30, max = -1e30;
  int i = 0;
  for(Level::const_patchIterator r = _level->patchesBegin();
	      r != _level->patchesEnd(); r++, i++ ){
    for(NodeIterator n = (*r)->getNodeIterator(); !n.done(); n++){
      min = SCICore::Math::Min( min, _vars[i][*n]);
      max = SCICore::Math::Max( max, _vars[i][*n]);
    }
  }
  if (min == max){
    min -= 1e-6;
    max += 1e-6;
  }
  data_min = double(min);
  data_max = double(max);
}

template <class T>
void NCScalarField<T>::compute_bounds()
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


template <class T>
void NCScalarField<T>::get_boundary_lines(Array1<Point>& lines)
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

template <class T>
int NCScalarField<T>::interpolate(const Point& p, double& value, int&,
                                   double epsilon1, double epsilon2,
                                   int) {
    return interpolate(p, value, epsilon1, epsilon2);
}

template <class T>
int NCScalarField<T>::interpolate(const Point& p, double& value, double eps,
                                   double)
{
  using SCICore::Math::Interpolate;

  Uintah::Box b;
  int i;
  Level::const_patchIterator r;
  for(i = 0, r = _level->patchesBegin();
      r != _level->patchesEnd(); r++, i++){
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


  if (ix<0) { if (x<-eps) return 0; else ix=lower.x(); }
    if (iy<0) { if (y<-eps) return 0; else iy=lower.y(); }
    if (iz<0) { if (z<-eps) return 0; else iz=lower.z(); }
    if (ix1>=upper.x()) { if (x>upper.x()-1+eps) return 0; else ix1=ix; }
    if (iy1>=upper.y()) { if (y>upper.y()-1+eps) return 0; else iy1=iy; }
    if (iz1>=upper.z()) { if (z>upper.z()-1+eps) return 0; else iz1=iz; }
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    typedef IntVector iv;
    double x00=Interpolate(_vars[i][iv(ix, iy, iz)], 
			   _vars[i][iv(ix1, iy, iz)], fx);
    double x01=Interpolate(_vars[i][iv(ix, iy, iz1)],
			   _vars[i][iv(ix1, iy, iz1)], fx);
    double x10=Interpolate(_vars[i][iv(ix, iy1, iz)],
			   _vars[i][iv(ix1, iy1, iz)], fx);
    double x11=Interpolate(_vars[i][iv(ix, iy1, iz1)],
			   _vars[i][iv(ix1, iy1, iz1)], fx);
    double y0=Interpolate(x00, x10, fy);
    double y1=Interpolate(x01, x11, fy);
    value=Interpolate(y0, y1, fz);
    return 1;
}


template <class T>
Vector NCScalarField<T>::gradient(const Point& p)
{
  using SCICore::Math::Interpolate;
  Uintah::Box b;
  int i;
  Level::const_patchIterator r;
  for(i = 0, r = _level->patchesBegin();
      r != _level->patchesEnd(); r++, i++){
    b = (*r)->getBox();
    if(b.contains(p)){
      break;
    }
  }

  if (i >= _vars.size())
    return Vector(0,0,0);
  
  IntVector index;
  if( !(*r)->findCell( p, index))
    return Vector(0,0,0);
  
  int ix = index.x();
  int iy = index.y();
  int iz = index.z();
  int ix1 = ix+1;
  int iy1 = iy+1;
  int iz1 = iz+1;
  IntVector upper = (*r)->getNodeHighIndex();
  IntVector lower = (*r)->getNodeLowIndex();
  if(ix1 >= upper.x() ) return Vector(0,0,0);
  if(iy1 >= upper.y() ) return Vector(0,0,0);
  if(iz1 >= upper.z() ) return Vector(0,0,0);
      
  Vector diag = b.upper() - b.lower();
  Vector pn = p - b.lower();
  double x = pn.x()*(upper.x() - lower.x()-1)/diag.x();
  double y = pn.y()*(upper.y() - lower.y()-1)/diag.y();
  double z = pn.z()*(upper.z() - lower.z()-1)/diag.z();
  
  double fx=x-ix;
  double fy=y-iy;
  double fz=z-iz;
  typedef IntVector iv;
  double z00=Interpolate(_vars[i][index], _vars[i][iv(ix,iy,iz1)], fz);
  double z01=Interpolate(_vars[i][iv(ix, iy1, iz)],
			 _vars[i][iv(ix, iy1, iz1)], fz);
  double z10=Interpolate(_vars[i][iv(ix1, iy, iz)],
			 _vars[i][iv(ix1, iy, iz1)], fz);
  double z11=Interpolate(_vars[i][iv(ix1, iy1, iz)],
			 _vars[i][iv(ix1, iy1, iz1)], fz);
  double yy0=Interpolate(z00, z01, fy);
  double yy1=Interpolate(z10, z11, fy);
  double dx=(yy1-yy0)*(upper.x()-1)/diagonal.x();
  double x00=Interpolate(_vars[i][iv(ix, iy, iz)],
			 _vars[i][iv(ix1, iy, iz)], fx);
  double x01=Interpolate(_vars[i][iv(ix, iy, iz1)],
			 _vars[i][iv(ix1, iy, iz1)], fx);
  double x10=Interpolate(_vars[i][iv(ix, iy1, iz)],
			 _vars[i][iv(ix1, iy1, iz)], fx);
  double x11=Interpolate(_vars[i][iv(ix, iy1, iz1)],
			 _vars[i][iv(ix1, iy1, iz1)], fx);
  double y0=Interpolate(x00, x10, fy);
  double y1=Interpolate(x01, x11, fy);
  double dz=(y1-y0)*(upper.z()-1)/diagonal.z();
  double z0=Interpolate(x00, x01, fz);
  double z1=Interpolate(x10, x11, fz);
  double dy=(z1-z0)*(upper.y()-1)/diagonal.y();
  return Vector(dx, dy, dz);
}  
  
} // Datatypes
} // SCICore
