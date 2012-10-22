/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/Datatypes/NCVectorField.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/LevelP.h>
#include <Core/Util/Handle.h>
#include <Core/Grid/Grid.h>

#include <Core/Geometry/Vector.h>
#include <Core/Math/MinMax.h>

using std::vector;

namespace Uintah {

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
    min = Min( min, (*r)->getBox().lower());
    max = Max( max, (*r)->getBox().upper());
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
using namespace SCIRun;
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

} // End namespace Uintah

