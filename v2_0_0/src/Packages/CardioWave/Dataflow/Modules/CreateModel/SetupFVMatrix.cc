/*
 *  SetupFVMatrix.cc:
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>

extern "C" {
#include <Packages/CardioWave/Core/Algorithms/Vulcan.h>
}

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE SetupFVMatrix : public Module {
  GuiDouble	sigx1_;
  GuiDouble	sigy1_;
  GuiDouble	sigz1_;
  GuiDouble	sigx2_;
  GuiDouble	sigy2_;
  GuiDouble	sigz2_;
  GuiString	sprfile_;
  GuiString	volumefile_;
  GuiString	visfile_;
  
public:
  SetupFVMatrix(GuiContext *context);
  virtual ~SetupFVMatrix();
  virtual void execute();
};


DECLARE_MAKER(SetupFVMatrix)


SetupFVMatrix::SetupFVMatrix(GuiContext *context)
  : Module("SetupFVMatrix", context, Source, "CreateModel", "CardioWave"),
    sigx1_(context->subVar("sigx1")),
    sigy1_(context->subVar("sigy1")),
    sigz1_(context->subVar("sigz1")),
    sigx2_(context->subVar("sigx2")),
    sigy2_(context->subVar("sigy2")),
    sigz2_(context->subVar("sigz2")),
    sprfile_(context->subVar("sprfile")),
    volumefile_(context->subVar("volumefile")),
    visfile_(context->subVar("visfile"))
  
{
}

SetupFVMatrix::~SetupFVMatrix(){
}

void SetupFVMatrix::execute(){
  double sigx1 = sigx1_.get();
  double sigy1 = sigy1_.get();
  double sigz1 = sigz1_.get();
  double sigx2 = sigx2_.get();
  double sigy2 = sigy2_.get();
  double sigz2 = sigz2_.get();
  string sprfile = sprfile_.get();
  string volumefile = volumefile_.get();
  string visfile = visfile_.get();
  
  // must find ports and have valid data on inputs
  FieldIPort *ifib1 = (FieldIPort*)get_iport("PrimaryFiberOrientation");
  if (!ifib1) {
    error("Unable to initialize iport 'PrimaryFiberOrientation'.");
    return;
  }
  
  FieldIPort *ifib2 = (FieldIPort*)get_iport("SecondaryFiberOrientation");
  if (!ifib2) {
    error("Unable to initialize iport 'SecondaryFiberOrientation'.");
    return;
  }
  
  FieldIPort *icell = (FieldIPort*)get_iport("CellType");
  if (!icell) {
    error("Unable to initialize iport 'CellType'.");
    return;
  }
  
  FieldOPort *omesh = (FieldOPort*)get_oport("ReorderedMesh");
  if (!omesh) {
    error("Unable to initialize oport 'ReorderedMesh'.");
    return;
  }

  FieldHandle cellTypeH;
  if (!icell->get(cellTypeH) || 
      !cellTypeH.get_rep())
    return;
  
  FieldHandle primFiberOrientH;
  if (!ifib1->get(primFiberOrientH) || 
      !primFiberOrientH.get_rep())
    return;
  
  FieldHandle secFiberOrientH;
  if (!ifib2->get(secFiberOrientH) || 
      !secFiberOrientH.get_rep())
    return;
  
#if 0
  if (primFiberOrientH->mesh()->generation != cellTypeH->mesh()->generation ||
      secFiberOrientH->mesh()->generation != cellTypeH->mesh()->generation)
  {
    error("Input fields have to have the same mesh.");
    return;
  }
#endif

  HexVolField<Vector> *fo1 = dynamic_cast<HexVolField<Vector> *>(primFiberOrientH.get_rep());
  HexVolField<Vector> *fo2 = dynamic_cast<HexVolField<Vector> *>(secFiberOrientH.get_rep());
  HexVolField<int> *ct = dynamic_cast<HexVolField<int> *>(cellTypeH.get_rep());

  if (!fo1) {
    error("PrimFiberOrientation field wasn't a HexVolField<Vector>.");
    return;
  }
  if (!fo2) {
    error("SecFiberOrientation field wasn't a HexVolField<Vector>.");
    return;
  }
  if (!ct) {
    error("CellType field wasn't a HexVolField<int>.");
    return;
  }

  
  HexVolMeshHandle m = fo1->get_typed_mesh();
  HexVolMesh::Node::size_type nnodes;
  HexVolMesh::Cell::size_type ncells;
  m->size(nnodes);
  m->size(ncells);

  MESH *mesh = new MESH;
  mesh->vtx = new VERTEX[nnodes];
  mesh->numvtx = nnodes;
  mesh->elements = new VOLUME_ELEMENT[ncells];
  mesh->numelement = ncells;
  
  // fill in mesh from SCIRun field

  HexVolMesh::Node::iterator nb, ne; m->begin(nb); m->end(ne);
  int cnt=0;
  Point p;

  while(nb != ne) {
    m->get_center(p, *nb);
    mesh->vtx[cnt].x = p.x();
    mesh->vtx[cnt].y = p.y();
    mesh->vtx[cnt].z = p.z();

    int ctIdx = ct->fdata()[cnt];
    Vector f1 = fo1->fdata()[*nb];
    Vector f2 = fo2->fdata()[*nb];
    Vector f3 = Cross(f1, f2);
    

    if (ctIdx == 0) {
      mesh->vtx[cnt].sxx = f1.x()*sigx1*f1.x() + 
                              f1.y()*sigy1*f1.y() + 
                              f1.z()*sigz1*f1.z();
      mesh->vtx[cnt].sxy = f1.x()*sigx1*f2.x() + 
                              f1.y()*sigy1*f2.y() + 
                              f1.z()*sigz1*f2.z();
      mesh->vtx[cnt].sxz = f1.x()*sigx1*f3.x() + 
                              f1.y()*sigy1*f3.y() + 
                              f1.z()*sigz1*f3.z();
      mesh->vtx[cnt].syy = f2.x()*sigx1*f2.x() + 
                              f2.y()*sigy1*f2.y() + 
                              f2.z()*sigz1*f2.z();
      mesh->vtx[cnt].syz = f2.x()*sigx1*f3.x() + 
                              f2.y()*sigy1*f3.y() + 
                              f2.z()*sigz1*f3.z();
      mesh->vtx[cnt].szz = f3.x()*sigx1*f3.x() + 
                              f3.y()*sigy1*f3.y() + 
                              f3.z()*sigz1*f3.z();
      mesh->vtx[cnt].volume=0;
    } else { // assuming type = 1
      mesh->vtx[cnt].sxx = f1.x()*sigx2*f1.x() + 
                              f1.y()*sigy2*f1.y() + 
                              f1.z()*sigz2*f1.z();
      mesh->vtx[cnt].sxy = f1.x()*sigx2*f2.x() + 
                              f1.y()*sigy2*f2.y() + 
                              f1.z()*sigz2*f2.z();
      mesh->vtx[cnt].sxz = f1.x()*sigx2*f3.x() + 
                              f1.y()*sigy2*f3.y() + 
                              f1.z()*sigz2*f3.z();
      mesh->vtx[cnt].syy = f2.x()*sigx2*f2.x() + 
                              f2.y()*sigy2*f2.y() + 
                              f2.z()*sigz2*f2.z();
      mesh->vtx[cnt].syz = f2.x()*sigx2*f3.x() + 
                              f2.y()*sigy2*f3.y() + 
                              f2.z()*sigz2*f3.z();
      mesh->vtx[cnt].szz = f3.x()*sigx2*f3.x() + 
                              f3.y()*sigy2*f3.y() + 
                              f3.z()*sigz2*f3.z();
      mesh->vtx[cnt].volume=0;
    }
    cnt++;
    ++nb;
  }
  
  HexVolMesh::Cell::iterator cb, ce; m->begin(cb); m->end(ce);
  HexVolMesh::Node::array_type nodes;
  cnt=0;

  while(cb != ce) {
    m->get_nodes(nodes, *cb);
    mesh->elements[cnt].vtx[0] = nodes[0];
    mesh->elements[cnt].vtx[1] = nodes[1];
    mesh->elements[cnt].vtx[2] = nodes[2];
    mesh->elements[cnt].vtx[3] = nodes[3];
    mesh->elements[cnt].vtx[4] = nodes[4];
    mesh->elements[cnt].vtx[5] = nodes[5];
    mesh->elements[cnt].vtx[6] = nodes[6];
    mesh->elements[cnt].vtx[7] = nodes[7];
    cnt++;
    ++cb;
  }
    
  compute_volumes(mesh, volumefile.c_str());
  compute_matrix(mesh, sprfile.c_str());
  dump_vis(mesh, visfile.c_str());

  omesh->send(ct);
}

} // End namespace CardioWave
