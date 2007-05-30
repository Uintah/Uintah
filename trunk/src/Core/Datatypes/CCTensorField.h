#ifndef SCI_project_CCTensorField_h
#define SCI_project_CCTensorField_h 1

#include <Core/Datatypes/TensorField.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/GridP.h>
#include <Core/Grid/LevelP.h>

#include <SCIRun/Core/Geometry/Point.h>
#include <SCIRun/Core/Geometry/IntVector.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

  using namespace SCIRun;
  using std::vector;
  
class CCTensorField: public TensorField {
public:
  int nx;
  int ny;
  int nz;
  CCTensorField();
  CCTensorField(const CCTensorField&);
  CCTensorField(GridP grid, LevelP level, string var, int mat,
		const vector< CCVariable<Matrix3> >& vars);
  virtual ~CCTensorField() {}
  virtual TensorField* clone();

  virtual void compute_bounds();
  virtual void get_boundary_lines(Array1<Point>& lines);
  void computeHighLowIndices();


  Matrix3 grid(int i, int j, int k);
  void SetGrid( GridP g ){ _grid = g; }
  void SetLevel( LevelP l){ _level = l; computeHighLowIndices(); }
  const LevelP GetLevel() { return _level; }
  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
  void AddVar( const CCVariable<Matrix3>& var);
private:
  vector< CCVariable<Matrix3> > _vars;
  GridP _grid;
  LevelP _level;
  string _varname;
  int _matIndex;
  IntVector low;
  IntVector high;
};
} // End namespace Uintah

#endif
