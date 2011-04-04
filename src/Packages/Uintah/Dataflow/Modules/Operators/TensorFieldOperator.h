/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef __DERIVE_TENSORFIELDOPERATOR_H__
#define __DERIVE_TENSORFIELDOPERATOR_H__

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

#include <Packages/Uintah/Dataflow/Modules/Operators/TensorOperatorFunctors.h>
#include <Packages/Uintah/Dataflow/Modules/Operators/UnaryFieldOperator.h>
#include <Packages/Uintah/Dataflow/Modules/Operators/OperatorThread.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>


/*************************************************************
 *  How to add a new function:
 *
 * 1. Add a new struct in TensorOperatorFunctors.h which has the desired
 *    operation in it.  Follow the template in the file.
 * 2. Add a new case statement in TensorFieldOperator::performOperation
 *    using the new functor that you created in #1.
 * 3. Add a corresponding case statement to TensorParticlesOperator.cc
 * 4. Edit ../../GUI/TensorOperator.tcl to include a new radiobutton for
 *    your new operator.  Make sure the value is the same as the one
 *    you selected for the case statement in #2.
 *    Be sure to change the name of the radiobutton to $w.calc.myoperation .
 *    and create a new method for your operation.  Add $w.calc.myoperation
 *    to the list of radiobuttons that are packed at the end.
 * 5. Add your new tcl method to the nested if statements following the
 *    radio buttons in the code.
 * 6. Add the body of the new method following the pattern of the others.
 * 7. Add a new ui method corresponding to your new function.  This is
 *    referenced in the method created in 6.  This will allow you to show
 *    in the user interface what operation you are calculating.
 */
namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;

class TensorFieldOperatorAlgo: 
    public DynamicAlgoBase, public UnaryFieldOperator 
{
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LVMesh::handle_type                 LVMeshHandle;
  typedef HexTrilinearLgn<double>             FDdoubleBasis;
  typedef ConstantBasis<double>               CFDdoubleBasis;

  typedef GenericField<LVMesh, CFDdoubleBasis, FData3d<double, LVMesh> > CDField;
  typedef GenericField<LVMesh, FDdoubleBasis,  FData3d<double, LVMesh> > LDField;
  virtual FieldHandle execute(FieldHandle tensorfh, int op) = 0;
  void set_values( int row, int column, int plane, double delta,
                           int calc_type, double nx, double ny, double nz,
                           double tx, double ty, double tz);

  static CompileInfoHandle get_compile_info(const SCIRun::TypeDescription *ftd);
protected:
  template<class TensorField, class ScalarField>
  void performOperation(TensorField* tensorField, 
                        ScalarField* scalarField, int op);

  int row_, column_, plane_, calc_type_;
  double delta_, nx_, ny_, nz_, tx_, ty_, tz_;
};

void
TensorFieldOperatorAlgo::set_values(int row, int column, int plane, 
                                    double delta, int calc_type, 
                                    double nx, double ny, double nz,
                                    double tx, double ty, double tz)
{
  row_ = row; column_ = column;
  plane_ = plane;
  delta_ = delta;
  calc_type_ = calc_type;
  nx_ = nx; ny_ = ny; nz_ = nz;
  tx_ = tx; ty_ = ty; tz_ = tz;
}

template<class TensorField>
class TensorFieldOperatorAlgoT: public TensorFieldOperatorAlgo
{
public:
  virtual FieldHandle execute(FieldHandle tensorfh, int op);
};

template<class TensorField>
FieldHandle
TensorFieldOperatorAlgoT<TensorField>::execute(FieldHandle tensorfh, int op)
{
  TensorField *tensorField = (TensorField *)(tensorfh.get_rep());
  typename TensorField::mesh_handle_type mh = tensorField->get_typed_mesh();
  mh.detach();
  typename TensorField::mesh_type *mesh = mh.get_rep();

  FieldHandle scalarField;
  if( tensorField->basis_order() == 0 ){
    CDField *sf = scinew CDField( mesh );
    performOperation( tensorField, sf, op );
    scalarField = sf;
  } else {
    LDField *sf = scinew LDField( mesh );
    performOperation( tensorField, sf, op );
    scalarField = sf;
  }
  
  scalarField->copy_properties( tensorField );
  string name;
  if(tensorField->get_property("name", name)){
    switch(op) {
    case 0: // extract element i,j
      scalarField->set_property("name",
                                string(name + ":" +
                                       to_string( row_) + 
                                       "," + to_string( column_)),
                                false);
      break;
    case 1: // extract eigen value
      scalarField->set_property("name", 
                                string(name +":eigen"), false);
      break;
    case 2: // extract pressure
      scalarField->set_property("name", 
                                string(name +":pressure"), false);
      break;
    case 3: // tensor stress
      scalarField->set_property("name", 
                                string(name +":equiv_stress"), false);
      break;
    case 4: // tensor stress
      scalarField->set_property("name",
                                string(name +":sheer_stress"), false);
      break;
    case 5: // tensor stress
      scalarField->set_property("name",
                                string(name +"NdotSigmadotT"), false);
      break;
    default:
      scalarField->set_property("name", name, true);
                                
    }
  }

  return scalarField;
}

template<class TensorField, class ScalarField>
void TensorFieldOperatorAlgo::performOperation(TensorField* tensorField,
                                               ScalarField* scalarField,
                                               int op)
{
  initField(tensorField, scalarField);

  switch(op) {
  case 0: // extract element
    computeScalars(tensorField, scalarField,
		   TensorElementExtractionOp(row_, column_));
    break;
  case 1: // 2D eigen-value/vector
    if (calc_type_ == 0) {
      // e1 - e2
      if (plane_ == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYOp());
      else if (plane_ == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZOp());
      else
	computeScalars(tensorField, scalarField, Eigen2DYZOp());
    }
    else {
      // cos(e1 - e2) / delta
      if (plane_ == 2)
	computeScalars(tensorField, scalarField, Eigen2DXYCosOp(delta_));
      else if (plane_ == 1)
	computeScalars(tensorField, scalarField, Eigen2DXZCosOp(delta_));
      else
	computeScalars(tensorField, scalarField, Eigen2DYZCosOp(delta_));
    }
    break;
  case 2: // pressure
    computeScalars(tensorField, scalarField, PressureOp());
    break;
  case 3: // equivalent stress 
    computeScalars(tensorField, scalarField, EquivalentStressOp());
    break;
  case 4: // Determinant
    computeScalars(tensorField, scalarField, Determinant());
    break;
  case 5: // n . sigma. t
    {
    computeScalars(tensorField, scalarField, NDotSigmaDotTOp(nx_, ny_, nz_,
                                                             tx_, ty_, tz_));
    }
    break;
  default:
    std::cerr << "TensorFieldOperator::performOperation: "
	      << "Unexpected Operation Type #: " << op << "\n";
  }
}

} // end namespace Uintah 

#endif // __DERIVE_TENSORFIELDOPERATOR_H__

