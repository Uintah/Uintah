/*
 *  CastTVtoMLV.cc
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/MaskedLatVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Geometry/BBox.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE CastTVtoMLV : public Module {
private:
  GuiInt nx_;
  GuiInt ny_;
  GuiInt nz_;
public:
  CastTVtoMLV(GuiContext* ctx);
  virtual ~CastTVtoMLV();
  virtual void execute();
};

  DECLARE_MAKER(CastTVtoMLV)

CastTVtoMLV::CastTVtoMLV(GuiContext* ctx)
  : Module("CastTVtoMLV", ctx, Filter, "FieldsGeometry", "SCIRun"),
    nx_(ctx->subVar("nx")), ny_(ctx->subVar("ny")), nz_(ctx->subVar("nz"))
{
}

CastTVtoMLV::~CastTVtoMLV(){
}

void CastTVtoMLV::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *iport_ = (FieldIPort*)get_iport("TetVolFieldMask");

  if (!iport_) {
    error("Unable to initialize iport 'TetVolFieldMask'.");
    return;
  }
  
  FieldOPort *oport_ = (FieldOPort*)get_oport("LatVolField");
  if (!oport_) {
    error("Unable to initialize oport 'LatVolField'.");
    return;
  }

  FieldHandle ifieldH;
  if (!iport_->get(ifieldH) || 
      !ifieldH.get_rep())
    return;
  
  // we expect that the input field is a TetVolField<Vector>
  if ( ifieldH.get_rep()->get_type_description()->get_name() ==
       "TetVolField<Vector>" )
  {
    error("Input volume is not a TetVolField<Vector>.");
    return;
  }                     

  TetVolField<Vector> *tv = (TetVolField<Vector> *) ifieldH.get_rep();
  TetVolMesh *tvm = (TetVolMesh *) tv->get_typed_mesh().get_rep();

  BBox b = tvm->get_bounding_box();

  // break up the volume into cells, with nx/ny/nz specified via the GUI
  int nx = nx_.get();
  int ny = ny_.get();
  int nz = nz_.get();

  MaskedLatVolMesh *mlvm = 
    scinew MaskedLatVolMesh(nx, ny, nz, b.min(), b.max());
  MaskedLatVolField<Vector> *lv = 
    scinew MaskedLatVolField<Vector>(mlvm, Field::NODE);

  // for each node in the LatVol, check to see if it's inside the TetMesh
  //    if it is, use the weights from get_weights and interpolate
  //    the fiber vectors

  MaskedLatVolMesh::Node::iterator ib, ie; mlvm->begin(ib); mlvm->end(ie);
  TetVolMesh::Cell::index_type tet;
  tvm->synchronize(Mesh::LOCATE_E); // for get_weights
  Point p;
  int cnt=0;
  while (ib != ie) {
    mlvm->get_center(p, *ib);
    if (tvm->locate(tet, p)) {
      cnt++;
      TetVolMesh::Node::array_type nodes;
      vector<double> weights;
      tvm->get_weights(p, nodes, weights);
      Vector f1(0,0,0);
      for (unsigned int i=0; i<nodes.size(); i++) {
	f1+=tv->fdata()[nodes[i]] * weights[i];
      }
      lv->fdata()[*ib]=f1;
    } else {
      mlvm->mask_cell(MaskedLatVolMesh::Cell::index_type(mlvm, ib.i_, ib.j_, ib.k_));
    }
    ++ib;
  }
  oport_->send(lv);
}

} // End namespace SCIRun
