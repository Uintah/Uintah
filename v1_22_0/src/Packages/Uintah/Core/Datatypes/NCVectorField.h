#ifndef SCI_project_NCVectorField_h
#define SCI_project_NCVectorField_h 1

#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>

#include <Core/Geometry/Point.h>
#include <Core/Datatypes/VectorFieldRG.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

using namespace SCIRun;
using std::vector;

class SCICORESHARE NCVectorField : public VectorFieldRG {
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
