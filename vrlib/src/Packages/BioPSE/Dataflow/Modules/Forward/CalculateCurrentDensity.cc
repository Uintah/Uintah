/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


/*
 *  CalculateCurrentDensity: Compute current through a volume
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Geometry/Vector.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class CalculateCurrentDensity : public Module {
typedef ConstantBasis<Vector>               ConVBasis;
typedef TetLinearLgn<int>                   TFDintBasis;
typedef TetLinearLgn<Tensor>                TFDTensorBasis;
typedef TetLinearLgn<Vector>                TFDVectorBasis;
typedef TetVolMesh<TetLinearLgn<Point> >    TVMesh;
typedef GenericField<TVMesh, TFDTensorBasis, vector<Tensor> > TVFieldT;
typedef GenericField<TVMesh, TFDVectorBasis, vector<Vector> > TVFieldV;
typedef GenericField<TVMesh, ConVBasis, vector<Vector> >      TVFieldCV;
typedef GenericField<TVMesh, TFDintBasis,    vector<int> >    TVFieldI;
public:
  CalculateCurrentDensity(GuiContext *context);
  virtual ~CalculateCurrentDensity();
  virtual void execute();
};

DECLARE_MAKER(CalculateCurrentDensity)


CalculateCurrentDensity::CalculateCurrentDensity(GuiContext *context)
  : Module("CalculateCurrentDensity", context, Filter, "Forward", "BioPSE")
{
}

CalculateCurrentDensity::~CalculateCurrentDensity()
{
}

void
CalculateCurrentDensity::execute()
{
  FieldHandle efieldH, sigmasH;
  if (!get_input_handle("TetMesh EField", efieldH)) return;
  if (!get_input_handle("TetMesh Sigmas", sigmasH)) return;

  if (efieldH->mesh().get_rep() != sigmasH->mesh().get_rep()) {
    error("EField and Sigma Field need to have the same mesh.");
    return;
  }

  TVFieldV *efield = dynamic_cast<TVFieldV*>(efieldH.get_rep());
  if (!efield) {
    error("EField isn't a TetVolField<Vector>.");
    return;
  }

  bool index_based = true;
  TVFieldI *sigmasInt =  dynamic_cast<TVFieldI*>(sigmasH.get_rep());
  TVFieldT *sigmasTensor = dynamic_cast<TVFieldT*>(sigmasH.get_rep());
  if (!sigmasInt && !sigmasTensor) {
    error("Sigmas isn't a TetVolField<Tensor> or TetVolField<int>.");
    return;
  }
  if (sigmasTensor) index_based = false;

  if (sigmasH->basis_order() != 0) {
    error("Need sigmas at Cells");
    return;
  }
  if (efieldH->basis_order() != 0) {
    error("Need efield at Cells");
    return;
  }

  vector<pair<string, Tensor> > conds;
  if (index_based && !sigmasH->get_property("conductivity_table", conds)) {
    error("No conductivity_table found in Sigmas.");
    return;
  }

  // For each cell in the mesh, find the dot product of the gradient
  // vector and the conductivity tensor.  The result is a vector field
  // with data at cells.

  // Create output mesh
  //  OFIELD *ofield = scinew OFIELD(imesh, 0);
  
  TVMesh::handle_type mesh = efield->get_typed_mesh();
  TVMesh::Cell::iterator fi, fe;
  mesh->begin(fi);
  mesh->end(fe);

  TVFieldCV *ofield = scinew TVFieldCV(mesh);

  while (fi != fe) {
    Vector vec;
    Vector e;
    efield->value(e, *fi);
    Tensor s;
    if (index_based) {
      int sigma_idx;
      sigmasInt->value(sigma_idx, *fi);
      s=conds[sigma_idx].second;
    } else {
      sigmasTensor->value(s, *fi);
    }
    // - sign added to vector to account for E = - Del V
    vec = Vector(-(s.mat_[0][0]*e.x()+s.mat_[0][1]*e.y()+s.mat_[0][2]*e.z()),
		 -(s.mat_[1][0]*e.x()+s.mat_[1][1]*e.y()+s.mat_[1][2]*e.z()),
		 -(s.mat_[2][0]*e.x()+s.mat_[2][1]*e.y()+s.mat_[2][2]*e.z()));

    ofield->set_value(vec,*fi);
    ++fi;
  }

  FieldHandle ftmp(ofield);
  send_output_handle("Currents", ftmp);
}

} // End namespace BioPSE
