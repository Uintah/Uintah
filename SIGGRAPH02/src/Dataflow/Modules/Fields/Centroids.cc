/*
 *  Centroids.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE Centroids : public Module {
public:
  Centroids(GuiContext* ctx);
  virtual ~Centroids();
  virtual void execute();
};

  DECLARE_MAKER(Centroids)

Centroids::Centroids(GuiContext* ctx)
  : Module("Centroids", ctx, Source, "Fields", "SCIRun")
{
}

Centroids::~Centroids(){
}

void Centroids::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *ifieldPort = (FieldIPort*)get_iport("TetVolField");

  if (!ifieldPort) {
    error("Unable to initialize iport 'TetVolField'.");
    return;
  }
  FieldHandle ifieldH;
  if (!ifieldPort->get(ifieldH) || !ifieldH.get_rep()) return;
  MeshHandle ifieldMeshH = ifieldH->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh*>(ifieldMeshH.get_rep());
  if (!tvm) {
    error("Input data wasn't a TetVolField.");
    return;
  }

  FieldOPort *ofieldPort = (FieldOPort*)get_oport("PointCloudField");
  if (!ofieldPort) {
    error("Unable to initialize oport 'PointCloudField'.");
    return;
  }

  PointCloudMeshHandle pcm(scinew PointCloudMesh);
  TetVolMesh::Cell::iterator ci, ce;
  tvm->begin(ci); tvm->end(ce);
  Point p;
  while (ci != ce) {
    tvm->get_center(p, *ci);
    pcm->add_node(p);
    ++ci;
  }
  
  FieldHandle ofieldH(scinew PointCloudField<double>(pcm, Field::NODE));
  ofieldPort->send(ofieldH);

}
} // End namespace SCIRun


