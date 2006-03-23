/*
 *  ReclassifyInteriorTets.cc:
 *
 *   Written by:
 *   Joe Tranquillo
 *   Duke University 
 *   Biomedical Engineering Department
 *   August 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/GenericField.h>

extern "C" {
#include <Packages/CardioWave/Core/Algorithms/Vulcan.h>
}

namespace CardioWave {

using namespace SCIRun;

class ReclassifyInteriorTets : public Module {
  GuiString	threshold_;
  GuiString     tag_;
public:
  ReclassifyInteriorTets(GuiContext *context);
  virtual ~ReclassifyInteriorTets();
  virtual void execute();
};


DECLARE_MAKER(ReclassifyInteriorTets)


ReclassifyInteriorTets::ReclassifyInteriorTets(GuiContext *context)
  : Module("ReclassifyInteriorTets", context, Source, "CreateModel", "CardioWave"),
    threshold_(context->subVar("threshold")),
    tag_(context->subVar("tag"))
{
}

ReclassifyInteriorTets::~ReclassifyInteriorTets(){
}

void ReclassifyInteriorTets::execute(){

  typedef TetVolMesh<TetLinearLgn<Point> > TVMesh;
  typedef LockingHandle<TVMesh> TVMeshHandle;
  typedef GenericField<TVMesh, TetLinearLgn<double>, vector<double> > TVField_double;
  typedef GenericField<TVMesh, TetLinearLgn<Vector>, vector<Vector> > TVField_Vector;
  typedef GenericField<TVMesh, ConstantBasis<Vector>, vector<Vector> > TVField_Vector_const;


  double threshold = atof(threshold_.get().c_str());

  // must find ports and have valid data on inputs
  FieldIPort *imesh = (FieldIPort*)get_iport("TetsIn");
  if (!imesh) {
    error("Unable to initialize iport 'TetsIn'.");
    return;
  }
  
  FieldOPort *omesh = (FieldOPort*)get_oport("TetsOut");
  if (!omesh) {
    error("Unable to initialize oport 'TetsOut'.");
    return;
  }

  FieldHandle meshH;
  if (!imesh->get(meshH) || 
      !meshH.get_rep())
    return;

  TVField_Vector *tv_old = dynamic_cast<TVField_Vector *>(meshH.get_rep());
  if (!tv_old) {
    error("Input field wasn't a TetVolField<Vector>.");
    return;
  }
  
  if (tv_old->basis_order() != 0) {
    error("Data must be at the cells for reclassification.");
    return;
  }

  TVMeshHandle mesh = tv_old->get_typed_mesh();

  TVMesh::Node::array_type nodes;
  Point centroid, p;
  TVMesh::Node::size_type nnodes;
  mesh->size(nnodes);
  TVMesh::Cell::size_type ncells;
  mesh->size(ncells);

  int tag = atoi(tag_.get().c_str());

  // copy the fdata for valid nodes
  TVField_Vector_const *tv_new = scinew TVField_Vector_const(mesh);

  // find the tets with centroid far from their nodes
  TVMesh::Cell::iterator cb, ce; mesh->begin(cb); mesh->end(ce);
  int i;
  while (cb!=ce) {
    mesh->get_nodes(nodes, *cb);
    mesh->get_center(centroid, *cb);
    for (i=0; i<4; i++) {
      mesh->get_center(p, nodes[i]);
      if ((p-centroid).length() > threshold) {
	tv_new->fdata()[*cb]=Vector(tag,tag,tag);
      }
    }
    ++cb;
  }
  omesh->send(tv_new);
}
} // End namespace CardioWave
