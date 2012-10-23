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

#ifndef SCI_project_NCVectorField_h
#define SCI_project_NCVectorField_h 1

#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <Core/Geometry/Point.h>
#include <Core/Datatypes/VectorFieldRG.h>

#include <vector>

namespace Uintah {

using namespace SCIRun;
using std::vector;

class NCVectorField : public VectorFieldRG {
public:
  NCVectorField();
  NCVectorField(const NCVectorField&);
  NCVectorField(GridP grid, LevelP level, string var, int mat,
		const vector< NCVariable<Vector> >& vars);
  virtual ~NCVectorField() {}
  virtual VectorField* clone();

  virtual void compute_bounds();
  virtual void get_boundary_lines(Array1<Point>& lines);
  virtual int interpolate(const Point&, Vector&);
  virtual int interpolate(const Point&, Vector&,
			  int& ix, int exhaustive=0);

  void SetGrid( GridP g ){ grid = g; }
  void SetLevel( LevelP l){ _level = l; }
  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
  void AddVar( const NCVariable<Vector>& var);
//   int nPatches(){ return _vars.size();}
//   string& name(){ return _varname; }
//   int material(){ return _matIndex; }
//   const NCVariable<Vector>& var(int i){ return _vars[i];}
private:
  vector< NCVariable<Vector> > _vars;
  GridP grid;
  LevelP _level;
  string _varname;
  int _matIndex;
};
} // End namespace Uintah


#endif
