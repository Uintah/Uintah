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
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/MaskedLatticeVol.h>
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
  CastTVtoMLV(const string& id);
  virtual ~CastTVtoMLV();
  virtual void execute();
};

extern "C" PSECORESHARE Module* make_CastTVtoMLV(const string& id) {
  return scinew CastTVtoMLV(id);
}

CastTVtoMLV::CastTVtoMLV(const string& id)
  : Module("CastTVtoMLV", id, Source, "Fields", "SCIRun"),
    nx_("nx", id, this), ny_("ny", id, this), nz_("nz", id, this)
{
}

CastTVtoMLV::~CastTVtoMLV(){
}

void CastTVtoMLV::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *iport_ = (FieldIPort*)get_iport("TetVolMask");

  if (!iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  
  FieldOPort *oport_ = (FieldOPort*)get_oport("LatticeVol");
  if (!oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle ifieldH;
  if (!iport_->get(ifieldH) || 
      !ifieldH.get_rep())
    return;
  
  // we expect that the input field is a TetVol<Vector>
  cout << "typename = " << ifieldH->get_type_description()->get_name() << '\n';
  if (ifieldH->get_type_description()->get_name() !=
      get_type_description((TetVol<Vector> *)0)->get_name())
  {
    postMessage("CastTVtoMLV: ERROR: input volume is not a TetVol<Vector>.  Exiting.");
    return;
  }                     

  TetVol<Vector> *tv = dynamic_cast<TetVol<Vector> *>(ifieldH.get_rep());
  TetVolMesh *tvm = tv->get_typed_mesh().get_rep();
  BBox b = tvm->get_bounding_box();

  // break up the volume into cells, with nx/ny/nz specified via the GUI
  int nx = nx_.get();
  int ny = ny_.get();
  int nz = nz_.get();

  LatVolMesh *lvm = scinew LatVolMesh(nx, ny, nz, b.min(), b.max());
  MaskedLatticeVol<Vector> *lv = 
    scinew MaskedLatticeVol<Vector>(lvm, Field::NODE);
  lv->initialize_mask(1);

  // for each node in the LatVol, check to see if it's inside the TetMesh
  //    if it is, use the weights from get_weights and interpolate
  //    the fiber vectors

  LatVolMesh::Node::iterator ib, ie; lvm->begin(ib); lvm->end(ie);
  TetVolMesh::Cell::index_type tet;
  Point p;
  int cnt=0;
  while (ib != ie) {
    lvm->get_center(p, *ib);
    if (tvm->locate(tet, p)) {
      cnt++;
      TetVolMesh::Node::array_type nodes;
      vector<double> weights;
      tvm->get_weights(p, nodes, weights);
      Vector f1(0,0,0);
      int i;
      for (i=0; i<nodes.size(); i++) {
	f1+=tv->fdata()[nodes[i]] * weights[i];
      }
      lv->fdata()[*ib]=f1;
    } else {
      lv->mask()[*ib] = 0;
    }
    ++ib;
  }
  cerr << "CastTVtoMLV: found "<<cnt<<" of "<<nx*ny*nz<<" nodes ("<<cnt*100./(nx*ny*nz)<<"%)\n";
  oport_->send(lv);
}

} // End namespace SCIRun
