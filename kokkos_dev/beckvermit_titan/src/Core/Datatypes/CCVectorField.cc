/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/Level.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Grid.h>
#include <Core/Datatypes/CCVectorField.h>

#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

#include <iostream>

using std::cerr;
using std::endl;
using std::vector;

namespace Uintah {

CCVectorField::CCVectorField()
  :VectorFieldRG()
{
}

CCVectorField::CCVectorField(const CCVectorField& copy)
  //  : UintahScalarField( copy )
  :VectorFieldRG( copy ), grid(copy.grid), level(copy.level),
    _varname(copy._varname), _matIndex(copy._matIndex)
{
  for(int i = 0; i < (int)copy._vars.size(); i++){
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
  for(int i = 0; i < (int)vars.size(); i++){
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
    min = Min( min, (*r)->getBox().lower());
    max = Max( max, (*r)->getBox().upper());
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
  using namespace SCIRun;

  int i;
  IntVector index;
  index = level->getCellIndex( p );
  Level::const_patchIterator r;
  for(i = 0, r = level->patchesBegin();
      r != level->patchesEnd(); r++, i++){
    if( (*r)->containsCell( index )){
      break;
    }
  }

  if (i >= (int)_vars.size() || r == level->patchesEnd() )
    return 0;
  
  value = _vars[i][index];
  return 1;
}

} // End namespace Uintah
