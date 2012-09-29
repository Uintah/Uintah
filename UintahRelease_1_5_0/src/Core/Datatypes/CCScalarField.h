/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef SCI_project_CCScalarField_h
#define SCI_project_CCScalarField_h 1

//#include "Packages/UintahScalarField.h"
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Variables/CellIterator.h>

#include <Core/Datatypes/ScalarFieldRGBase.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Point.h>

#include <sci_values.h>
#include <vector>

namespace Uintah {

  using namespace SCIRun;

template <class T>
class CCScalarField : public ScalarFieldRGBase {
public:
  CCScalarField();
  CCScalarField(const CCScalarField<T>&);
  CCScalarField(GridP grid, LevelP level, std::string var, int mat,
		const std::vector< CCVariable<T> >& vars);
  virtual ~CCScalarField() {}
  virtual ScalarField* clone();

  virtual void compute_bounds();
  virtual void compute_minmax();
  virtual void get_boundary_lines(Array1<Point>& lines);
  virtual Vector gradient(const Point&);
  virtual int interpolate(const Point&, double&,
			  double epsilon1=1.e-6,
			  double epsilon2=1.e-6);

  virtual int interpolate(const Point&, double&,
			  int& ix, double epsilon1=1.e-6,
			  double epsilon2=1.e-6, int exhaustive=0);

  T grid(int i, int j, int k);
  virtual double get_value( int i, int j, int k);
  void computeHighLowIndices();

  void SetGrid( GridP g ){ _grid = g; }
  void SetLevel( LevelP l){ _level = l; }
  void SetName( std::string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
  void AddVar( const CCVariable<T>& var,  const Patch* p);

private:
  std::vector< CCVariable<T> > _vars;
  GridP _grid;
  LevelP _level;
  std::string _varname;
  int _matIndex;
  IntVector high;
  IntVector low;
};

template <class T>
CCScalarField<T>::CCScalarField()
  :ScalarFieldRGBase(),
   high(-INT_MAX,-INT_MAX,-INT_MAX),
   low(INT_MAX,INT_MAX,INT_MAX)
{
}

template <class T>
CCScalarField<T>::CCScalarField(const CCScalarField<T>& copy)
  :ScalarFieldRGBase( copy ), _grid(copy._grid), _level(copy._level),
    _varname(copy._varname), _matIndex(copy._matIndex), 
   high(-INT_MAX,-INT_MAX,-INT_MAX),
   low(INT_MAX,INT_MAX,INT_MAX)
{
  for(int i = 0; i < (int)copy._vars.size(); i++){
    _vars.push_back( copy._vars[i] );
  }
  computeHighLowIndices();
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
}

template <class T>
CCScalarField<T>::CCScalarField(GridP grid, LevelP level,
				std::string var, int mat,
				const std::vector< CCVariable<T> >& vars)
  : ScalarFieldRGBase(), _grid(grid), _level(level),
    _varname(var), _matIndex(mat),
   high(-INT_MAX,-INT_MAX,-INT_MAX),
   low(INT_MAX,INT_MAX,INT_MAX)
{
  for(int i = 0; i < (int)vars.size(); i++){
    _vars.push_back( vars[i]);
  }
  computeHighLowIndices();
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
}

template <class T>
void CCScalarField<T>::computeHighLowIndices()
{
  for(Level::const_patchIterator r = _level->patchesBegin();
      r != _level->patchesEnd(); r++){
    low = Min( low, (*r)->getCellLowIndex());
    high = Max( high, (*r)->getCellHighIndex());
  }
}

template <class T>
T CCScalarField<T>::grid(int i, int j, int k)
{
  static const Patch* patch = 0;
  static int ii = 0;
  IntVector id(i,j,k);
  id = low + id;

  if( patch !=0 && patch->containsNode(id)) {
    return _vars[ii][id];
  } else {
    ii = 0;
    for(Level::const_patchIterator r = _level->patchesBegin();
	r != _level->patchesEnd(); r++, ii++){
      if( (*r)->containsCell( id ) ){
	patch = (*r);
	return _vars[ii][id];
      }
    }
  }
  patch = 0;
  return 0;
}

template <class T>
double  CCScalarField<T>::get_value(int i, int j, int k)
{
  return double( grid(i,j,k) );
}

template <class T>
void CCScalarField<T>::AddVar( const CCVariable<T>& v, const Patch* p)
{
  _vars.push_back( v );
  low = Min( low, p->getCellLowIndex());
  high = Max( high, p->getCellHighIndex());
  nx = high.x() - low.x();
  ny = high.y() - low.y();
  nz = high.z() - low.z();
}

template<class T>
ScalarField*  CCScalarField<T>::clone()
{
  return scinew CCScalarField<T>(*this);
} 

template<class T>
void CCScalarField<T>::compute_minmax()
{
  T min = T(SHRT_MAX), max = T(-SHRT_MAX);
  int i = 0;
  for(Level::const_patchIterator r = _level->patchesBegin();
	      r != _level->patchesEnd(); r++, i++ ){
    for(CellIterator n = (*r)->getCellIterator(); !n.done(); n++){
      min = Min( min, _vars[i][*n]);
      max = Max( max, _vars[i][*n]);
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
void CCScalarField<T>::compute_bounds()
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

template <class T>
void CCScalarField<T>::get_boundary_lines(Array1<Point>& lines)
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
int CCScalarField<T>::interpolate(const Point& p, double& value, int&,
                                   double epsilon1, double epsilon2,
                                   int) {
    return interpolate(p, value, epsilon1, epsilon2);
}

template <class T>
int CCScalarField<T>::interpolate(const Point& p, double& value, double,
                                   double)
{
  static const Patch* patch = 0;
  static int i = 0;
  //  static int count = 0;
  IntVector index;
  if( patch !=0 && patch->findCell(p, index)){
    value = _vars[i][index];
    return 1;
  } else {

    Level::const_patchIterator r;
    for(i = 0, r = _level->patchesBegin();
	r != _level->patchesEnd(); r++, i++){
     
      
      if( (*r)->findCell(p, index)){
	patch = *r;
	value = _vars[i][index];

	return 1;
      }
    }
    if (i >= (int)_vars.size() || r == _level->patchesEnd() ){
      patch = 0;
      return 0;
    }
    
  }
  patch = 0;
  return 0;
}


template <class T>
Vector CCScalarField<T>::gradient(const Point& p)
{
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
  
  if (i >= (int)_vars.size())
    return Vector(0,0,0);
  
  IntVector index;
  if( !(*r)->findCell( p, index))
    return Vector(0,0,0);
  
  int ix = index.x();
  int iy = index.y();
  int iz = index.z();

  IntVector upper = (*r)->getCellHighIndex();
  IntVector lower = (*r)->getCellLowIndex();
  
  Vector diag = b.upper() - b.lower();
  //Vector pn = p - b.lower();
  int nx = upper.x() - lower.x();
  int ny = upper.y() - lower.y();
  int nz = upper.z() - lower.z();
  double dx = diag.x()/(nx-1);
  double dy = diag.y()/(ny-1);
  double dz = diag.z()/(nz-1);
  
  typedef IntVector iv;
				
  // compute gradients
  double gradX, gradY, gradZ;

  // compute the X component
  if (ix == 0){			// point is in left face cell
    gradX = ( -3.0 * _vars[i][iv(ix, iy, iz)] + 4.0*_vars[i][iv(ix+1,iy,iz)]
	      - _vars[i][iv(ix+2,iy,iz)])/ dx;
  } else if( ix == nx - 1){	// point is in right face cell
    gradX = ( 3.0 * _vars[i][iv(ix, iy, iz)] - 4.0*_vars[i][iv(ix-1,iy,iz)]
	      - _vars[i][iv(ix-2,iy,iz)])/ dx;
  } else {			// point is NOT in left or right face cell
    gradX = (_vars[i][iv(ix+1, iy, iz)] - _vars[i][iv(ix-1, iy, iz)])/(2.0 * dx );
  }
  // compute the Y component
  if (iy == 0){			// point is in bottom face cell
    gradY = ( -3.0 * _vars[i][iv(ix, iy, iz)] + 4.0*_vars[i][iv(ix,iy+1,iz)]
	      - _vars[i][iv(ix,iy+2,iz)])/ dy;
  } else if( iy == ny - 1){	// point is in top face cell
    gradY = ( 3.0 * _vars[i][iv(ix, iy, iz)] - 4.0*_vars[i][iv(ix,iy-1,iz)]
	      - _vars[i][iv(ix,iy-2,iz)])/ dy;
  } else {			// point is NOT in top or bottom face cell
    gradY = (_vars[i][iv(ix, iy+1, iz)] - _vars[i][iv(ix, iy-1, iz)])/(2.0 * dy );
  }
  // compute the Z component
  if (iz == 0){			// point is in a back face cell
    gradZ = ( -3.0 * _vars[i][iv(ix, iy, iz)] + 4.0*_vars[i][iv(ix,iy,iz+1)]
	      - _vars[i][iv(ix,iy,iz+2)])/ dz;
  } else if( iz == nz - 1){	// point is in a front face cell
    gradZ = ( 3.0 * _vars[i][iv(ix, iy, iz)] - 4.0*_vars[i][iv(ix,iy,iz-1)]
	      - _vars[i][iv(ix,iy,iz-2)])/ dz;
  } else {			//point is NOT in a front or back face cell
    gradZ = (_vars[i][iv(ix, iy, iz+1)] - _vars[i][iv(ix, iy, iz-1)])/(2.0 * dz );
  }
  
  return Vector( gradX, gradY, gradZ );

}  

} // End namespace Uintah

#endif
