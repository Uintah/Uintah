/*
 *  LatticeToHex.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE LatticeToHex : public Module {
private:
  int            last_gen_;
  FieldHandle    ofieldH_;
public:
  LatticeToHex(GuiContext* ctx);
  virtual ~LatticeToHex();
  virtual void execute();
};

  DECLARE_MAKER(LatticeToHex(GuiContext* ctx));

LatticeToHex::LatticeToHex(GuiContext* ctx)
  : Module("LatticeToHex", ctx, Source, "Fields", "SCIRun"), last_gen_(-1)
{
}

LatticeToHex::~LatticeToHex(){
}

void LatticeToHex::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *iport_ = (FieldIPort*)get_iport("Cell centered volume");

  if (!iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  
  FieldOPort *oport_ = (FieldOPort*)get_oport("Node centered volume");
  if (!oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle ifieldH;
  if (!iport_->get(ifieldH) || 
      !ifieldH.get_rep())
    return;
  
  if (ifieldH->generation == last_gen_) {
    oport_->send(ofieldH_);
    return;
  }
  last_gen_ = ifieldH->generation;

  // we expect that the input field is a TetVolField<Vector>
  if (ifieldH->get_type_description()->get_name() !=
      get_type_description((LatVolField<int> *)0)->get_name())
  {
    postMessage("LatticeToHex: ERROR: input volume is not a LatVolField<int>.  Exiting.");
    return;
  }                     

  if (ifieldH->data_at() != Field::NODE) {
    postMessage("LatticeToHex: ERROR: input volume data isn't node-centered.  Existing.");
    return;
  }                         

  LatVolField<int> *lv = dynamic_cast<LatVolField<int> *>(ifieldH.get_rep());
  LatVolMesh *lvm = lv->get_typed_mesh().get_rep();
  HexVolMesh *hvm = scinew HexVolMesh;

  // fill in the nodes and connectivities
  Point min = lvm->get_min();
  Vector diag = lvm->diagonal();
  int nx = lvm->get_nx();
  int ny = lvm->get_ny();
  int nz = lvm->get_nz();
  double dx = diag.x()/(nx-1);
  double dy = diag.y()/(ny-1);
  double dz = diag.z()/(nz-1);

  int i, j, k;
  for (k=0; k<nz; k++)
    for (j=0; j<ny; j++)
      for (i=0; i<nx; i++)
	hvm->add_point(min + Vector(dx*i, dy*j, dz*k));
  for (k=0; k<nz-1; k++)
    for (j=0; j<ny-1; j++)
      for (i=0; i<nx-1; i++) {
	HexVolMesh::Node::index_type n000, n001, n010, n011, n100, n101, n110, n111;
	n000 = (k  )*(nx*ny) + (j  )*nx + (i  );
	n001 = (k  )*(nx*ny) + (j  )*nx + (i+1);
	n010 = (k  )*(nx*ny) + (j+1)*nx + (i  );
	n011 = (k  )*(nx*ny) + (j+1)*nx + (i+1);
	n100 = (k+1)*(nx*ny) + (j  )*nx + (i  );
	n101 = (k+1)*(nx*ny) + (j  )*nx + (i+1);
	n110 = (k+1)*(nx*ny) + (j+1)*nx + (i  );
	n111 = (k+1)*(nx*ny) + (j+1)*nx + (i+1);
	hvm->add_hex(n000, n001, n011, n010, n100, n101, n111, n110);
      }

  HexVolField<int> *hv = scinew HexVolField<int>(hvm, Field::NODE);

  int c=0;
  for (k=0; k<nz-1; k++)
    for (j=0; j<ny-1; j++)
      for (i=0; i<nx-1; i++, c++)
	hv->fdata()[c] = lv->fdata()(k,j,i);

  ofieldH_ = hv;
  oport_->send(ofieldH_);
}

} // End namespace SCIRun


