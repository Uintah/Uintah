#ifndef SCI_project_NCScalarField_h
#define SCI_project_NCScalarField_h 1

#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/IntVector.h>
#include <vector>

namespace SCICore {
  namespace Datatypes {

using namespace SCICore::Geometry;
using namespace Uintah;
using std::vector;

template <class T>
class SCICORESHARE NCScalarField : public ScalarFieldRGBase {
public:
  NCScalarField();
  NCScalarField(const NCScalarField<T>&);
  NCScalarField(GridP grid, LevelP level, string var, int mat,
		const vector< NCVariable<T> >& vars);
  virtual ~NCScalarField() {}
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

//   virtual UintahScalarField::Rep getType(){ return UintahScalarField::NC;}

  void SetGrid( GridP g ){ _grid = g; }
  void SetLevel( LevelP l){ _level = l; }
  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
  void AddVar( const NCVariable<T>& var, const Patch* p);
//   int nPatches(){ return _vars.size();}
//   string& name(){ return _varname; }
//   int material(){ return _matIndex; }
//   const NCVariable<T>& var(int i){ return _vars[i];}
private:
  vector< NCVariable<T> > _vars;
  GridP _grid;
  LevelP _level;
  string _varname;
  int _matIndex;
  IntVector low;
  IntVector high;
};

} // End namespace Datatypes
} // End namespace SCICore

#endif
