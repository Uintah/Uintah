/*
 *  ExtractSurfNormals.cc:
 *
 *  Written by: Martin Cole
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Packages/MC/share/share.h>

namespace MC {

using namespace SCIRun;

class MCSHARE ExtractSurfNormals : public Module {
public:
  typedef   LockingHandle<PointCloudField<Vector> > pcv_t;
  ExtractSurfNormals(const string& id);

  virtual ~ExtractSurfNormals();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
private:
  //! extract normals, and fill out_
  void extract_normals(TriSurfMesh *tmesh);

  FieldIPort                 *ifld_;
  FieldOPort                 *ofld1_;
  pcv_t                       out1_;
};

extern "C" MCSHARE Module* make_ExtractSurfNormals(const string& id) {
  return scinew ExtractSurfNormals(id);
}

ExtractSurfNormals::ExtractSurfNormals(const string& id) : 
  Module("ExtractSurfNormals", id, Source, "Field", "MC"),
  out1_(scinew PointCloudField<Vector>)
{
  // Create the input port
  ifld_=scinew FieldIPort(this, "Field with normals (TriSurfFieldace)", 
			  FieldIPort::Atomic);
  add_iport(ifld_);

  // Create the output ports
  ofld1_=scinew FieldOPort(this, 
			  "vertex normals as vector field (PointCloudField)", 
			  FieldIPort::Atomic);
  add_oport(ofld1_);
}

ExtractSurfNormals::~ExtractSurfNormals(){
}

void ExtractSurfNormals::execute()
{
  FieldHandle input;
  if (!ifld_->get(input)) return;
  if (!input.get_rep()) {
    error("ExtractSurfNormals Error: No input data.");
    return;
  }
  MeshHandle mesh = input->mesh();
  
  // must be trisurface
  TriSurfMesh *tmesh = 0;
  tmesh = dynamic_cast<TriSurfMesh*>(mesh.get_rep());
  if (! tmesh) {
    error("ExtractSurfNormals Error: Must have TriSurfMesh.");
    return; 
  }

  // build the vector fields.
  extract_normals(tmesh);

  ofld1_->send(FieldHandle(out1_.get_rep()));
}

void 
ExtractSurfNormals::extract_normals(TriSurfMesh *mesh)
{
  PointCloudMeshHandle pcmesh1 = out1_->get_typed_mesh();
  vector<Vector> &fdata1 = out1_->fdata();

  // pass: over the nodes
  TriSurfMesh::Node::iterator niter; mesh->begin(niter);  
  TriSurfMesh::Node::iterator niter_end; mesh->end(niter_end);
  while (niter != niter_end) {
    Point p;
    mesh->get_point(p, *niter);
    pcmesh1->add_point(p);
    Vector v;
    mesh->get_normal(v, *niter);
    fdata1.push_back(v);
    ++niter;
  }

}


void ExtractSurfNormals::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace MC


