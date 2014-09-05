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
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/Vector.h>

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE VectorFieldsToTensorField : public Module {
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

  FieldOPort *otfld = (FieldOPort*)get_oport("Tensors");
  if (!otfld) {
    error("Unable to initialize oport 'Tensors'.");
    return;
  }

  LatVolField<Vector> *ev1 = 
    dynamic_cast<LatVolField<Vector> *>(ev1H.get_rep());
  if (!ev1) {
    error("Major eigenvector field isn't a LatVolField<Vector>.");
    return;
  }

  LatVolField<Vector> *ev2 = 
    dynamic_cast<LatVolField<Vector> *>(ev2H.get_rep());
  if (!ev2) {
    error("Median eigenvector field isn't a LatVolField<Vector>.");
    return;
  }

  LatVolMeshHandle ev1mesh = ev1->get_typed_mesh();
  LatVolMeshHandle ev2mesh = ev1->get_typed_mesh();
  if (ev1->data_at() != ev2->data_at()) {
    error("Vector fields must have the same data_at.");
    return;
  }

  if (ev1mesh->get_nx() != ev2mesh->get_nx() ||
      ev1mesh->get_ny() != ev2mesh->get_ny() ||
      ev1mesh->get_nz() != ev2mesh->get_nz())
  {
    error("Fields must be the same size.");
    return;
  }

  LatVolField<int> *tfield = scinew LatVolField<int>(ev1mesh, ev1->data_at());

  Vector *v1 = ev1->fdata().begin();
  Vector *v2 = ev2->fdata().begin();
  int *tidx = tfield->fdata().begin();
  vector<pair<string, Tensor> > conds;
  Tensor t(Vector(0,0,0), Vector(0,0,0), Vector(0,0,0));
  pair<string, Tensor> cond;
  cond.first = "-";
  cond.second = t;
  conds.push_back(cond);  // bath

  t=Tensor(Vector(1,0,0), Vector(0,1,0), Vector(1,1,1));
  conds.push_back(cond);  // ventricles

  while (v1 != ev1->fdata().end()) {
    if (v1->length() != 0) {
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
  }
  tfield->set_property("conductivity_tensors", conds, false);
  otfld->send(tfield);
}
} // End namespace CardioWave
