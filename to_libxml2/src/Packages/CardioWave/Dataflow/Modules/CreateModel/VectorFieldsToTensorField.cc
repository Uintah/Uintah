/*
 *  VectorFieldsToTensorField.cc:
 *
 *   Written by:
 *   David Weinstein
 *   May 2002
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Geometry/Vector.h>

namespace CardioWave {

using namespace SCIRun;

class VectorFieldsToTensorField : public Module {
public:
  VectorFieldsToTensorField(GuiContext *context);
  virtual ~VectorFieldsToTensorField();
  virtual void execute();
};


DECLARE_MAKER(VectorFieldsToTensorField)


VectorFieldsToTensorField::VectorFieldsToTensorField(GuiContext *context)
  : Module("VectorFieldsToTensorField", context, Source, 
	   "CreateModel", "CardioWave")
{
}


VectorFieldsToTensorField::~VectorFieldsToTensorField()
{
}

void
VectorFieldsToTensorField::execute()
{

  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef LockingHandle<LVMesh> LVMeshHandle;
  typedef GenericField<LVMesh, HexTrilinearLgn<int>, FData3d<int, LVMesh> > LVField_int;
  typedef GenericField<LVMesh, ConstantBasis<int>, FData3d<int, LVMesh> > LVField_int_const;
  typedef GenericField<LVMesh, HexTrilinearLgn<double>, FData3d<double, LVMesh> > LVField_double;
  typedef GenericField<LVMesh, HexTrilinearLgn<Vector>, FData3d<Vector, LVMesh> > LVField_Vector;
  
  FieldIPort *iev1 = (FieldIPort*)get_iport("Major Eigenvectors");
  if (!iev1) {
    error("Unable to initialize iport 'Major Eigenvectors'.");
    return;
  }
  FieldHandle ev1H;
  if (!iev1->get(ev1H) || !ev1H.get_rep()) {
    error("No valid major eigenvector field.");
    return;
  }

  FieldIPort *iev2 = (FieldIPort*)get_iport("Median Eigenvectors");
  if (!iev2) {
    error("Unable to initialize iport 'Median Eigenvectors'.");
    return;
  }
  FieldHandle ev2H;
  if (!iev2->get(ev2H) || !ev2H.get_rep()) {
    error("No valid median eigenvector field.");
    return;
  }

  FieldIPort *imask = (FieldIPort*)get_iport("Mask");
  if (!imask) {
    error("Unable to initialize iport 'Mask'.");
    return;
  }
  FieldHandle maskH;
  imask->get(maskH);

  FieldOPort *otfld = (FieldOPort*)get_oport("Tensors");
  if (!otfld) {
    error("Unable to initialize oport 'Tensors'.");
    return;
  }

  LVField_Vector *ev1 = 
    dynamic_cast<LVField_Vector *>(ev1H.get_rep());
  if (!ev1) {
    error("Major eigenvector field isn't a LatVolField<Vector>.");
    return;
  }

  LVField_Vector *ev2 = 
    dynamic_cast<LVField_Vector *>(ev2H.get_rep());
  if (!ev2) {
    error("Median eigenvector field isn't a LatVolField<Vector>.");
    return;
  }

  LVField_double *mask = 0;
  if (maskH.get_rep())
    mask = dynamic_cast<LVField_double *>(maskH.get_rep());
    
  LVMeshHandle ev1mesh = ev1->get_typed_mesh();
  LVMeshHandle ev2mesh = ev2->get_typed_mesh();
  LVMeshHandle maskmesh;
  if (mask) maskmesh = mask->get_typed_mesh();

  if (ev1->basis_order() != ev2->basis_order()) {
    error("Vector fields must have the same data_at.");
    return;
  }

  if (ev1mesh->get_ni() != ev2mesh->get_ni() ||
      ev1mesh->get_nj() != ev2mesh->get_nj() ||
      ev1mesh->get_nk() != ev2mesh->get_nk())
  {
    error("Fields must be the same size.");
    return;
  }

  if (mask) {
    if (ev1->basis_order() != mask->basis_order()) {
      error("Vector fields and mask must have the same data_at.");
      return;
    }
    if (ev1mesh->get_ni() != maskmesh->get_ni() ||
	ev1mesh->get_nj() != maskmesh->get_nj() ||
	ev1mesh->get_nk() != maskmesh->get_nk()) 
    {
      error("Fields must all be the same size.");
      return;
    }
  }

  int *tidx;
  FieldHandle ofield;
  
  if (ev1->basis_order() == 0)
  {
    LVField_int_const *tfield = scinew LVField_int_const(ev1mesh);
    ofield = dynamic_cast<Field *>(tfield);
    tidx = tfield->fdata().begin();
  }
  else if (ev1->basis_order() == 1)
  {
    LVField_int *tfield = scinew LVField_int(ev1mesh);
    ofield = dynamic_cast<Field *>(tfield);
    tidx = tfield->fdata().begin();  
  }
  else
  {
    error("Higher order elements are not supported");
    return;
  }
  
  Vector *v1 = ev1->fdata().begin();
  Vector *v2 = ev2->fdata().begin();
  double *maskiter=0;
  if (mask) maskiter = mask->fdata().begin();


  vector<pair<string, Tensor> > conds;
  Tensor t(Vector(0,0,0), Vector(0,0,0), Vector(0,0,0));
  pair<string, Tensor> cond;
  cond.first = "-";
  cond.second = t;
  conds.push_back(cond);  // bath

  t=Tensor(Vector(1,0,0), Vector(0,1,0), Vector(1,1,1));
  conds.push_back(cond);  // ventricles

  while (v1 != ev1->fdata().end()) {
    if (v1->length() != 0 && mask && (*maskiter != 0)) {

      Vector v3(Cross(*v1, *v2).normal());
      v3 *= v2->length();
      t = Tensor(*v1, *v2, v3);
      cond.second = t;
      *tidx = (int)conds.size();
      conds.push_back(cond);
    } else {
      *tidx = 0;
    }
    ++v1;
    ++v2;
    ++tidx;
    if (mask) ++maskiter;
  }
  ofield->set_property("conductivity_table", conds, false);
  otfld->send(ofield);
}
} // End namespace CardioWave
