
/*
 *  InsertVoltageSource: Insert a voltage source
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

// check for same meshes as last time -- if so, reuse contrib array
// fix have_some -- just need to know which TVM indices we've visited

#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/BioPSE/Dataflow/Modules/Forward/InsertVoltageSource.h>
#include <iostream>
#include <sstream>

namespace BioPSE {

using namespace SCIRun;

class InsertVoltageSource : public Module {
  GuiInt outside_;
  GuiInt groundfirst_;
public:
  InsertVoltageSource(GuiContext *context);
  virtual ~InsertVoltageSource();
  virtual void execute();
};


DECLARE_MAKER(InsertVoltageSource)


InsertVoltageSource::InsertVoltageSource(GuiContext *context)
  : Module("InsertVoltageSource", context, Filter, "Forward", "BioPSE"),
    outside_(context->subVar("outside")),
    groundfirst_(context->subVar("groundfirst"))
{
}

InsertVoltageSource::~InsertVoltageSource()
{
}

void InsertVoltageSource::execute() {
  FieldIPort* imesh = (FieldIPort *) get_iport("TetMesh");
  FieldIPort* isource = (FieldIPort *) get_iport("VoltageSource");
  FieldOPort* omesh = (FieldOPort *) get_oport("TetMesh");
  if (!imesh) {
    error("Unable to initialize iport 'TetMesh'.");
    return;
  }
  if (!isource) {
    error("Unable to initialize iport 'VoltageSource'.");
    return;
  }
  if (!omesh) {
    error("Unable to initialize oport 'TetMesh'.");
    return;
  }

  FieldHandle imeshH;
  if (!imesh->get(imeshH))
    return;
  if (!imeshH.get_rep()) {
    error("Empty input mesh.");
    return;
  }

  MeshHandle tetVolH = imeshH->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(tetVolH.get_rep());
  if (!tvm) {
    error("Input FEM wasn't a TetVolField.");
    return;
  }

  FieldHandle isourceH;
  if (!isource->get(isourceH))
    return;
  if (!isourceH.get_rep()) {
    error("Empty input source.");
    return;
  }

  int groundfirst = groundfirst_.get();
  vector<Point> sources;
  vector<double> vals;
  if (groundfirst) {
    // just need to know the position of the first point of the mesh
    const TypeDescription *meshtd = isourceH->mesh()->get_type_description();
    CompileInfoHandle ci =
      InsertVoltageSourceGetPtBase::get_compile_info(meshtd);
    Handle<InsertVoltageSourceGetPtBase> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    Point pt = algo->execute(isourceH->mesh());
    sources.push_back(pt);
    vals.push_back(0);
  } else {
    const TypeDescription *field_td = isourceH->get_type_description();
    const TypeDescription *loc_td = isourceH->data_at_type_description();
    CompileInfoHandle ci = 
      InsertVoltageSourceGetPtsAndValsBase::get_compile_info(field_td, 
							     loc_td);
    Handle<InsertVoltageSourceGetPtsAndValsBase> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->execute(isourceH, sources, vals);
  }

  vector<pair<int, double> > dirichlet;
  imeshH->get_property("dirichlet", dirichlet);

  vector<pair<string, Tensor> > conds;
  imeshH->get_property("conductivity_table", conds);

  // get our own local copy of the Field and mesh
  imeshH.detach();  

  int outside = outside_.get();

  TetVolMesh::Node::size_type tvm_nnodes;
  TetVolMesh::Cell::size_type tvm_ncells;
  tvm->size(tvm_nnodes);
  tvm->size(tvm_ncells);
  tvm->synchronize(Mesh::LOCATE_E);

  TetVolMesh::Node::array_type nbrs(4);

  Array1<vector<pair<double, double> > > closest(tvm_nnodes); 
                                     // store the dist/val
                                     // to source nodes
  Array1<int> have_some(tvm_nnodes);
  have_some.initialize(0);
  Array1<TetVolMesh::Node::index_type> bc_tet_nodes;

  for (unsigned int di=0; di<dirichlet.size(); di++) {
    int didx=dirichlet[di].first;
    // so other BCs can't over-write this one
    have_some[didx]=1;
    closest[didx].push_back(pair<double, double>(0,dirichlet[di].second));
  }
    
  // for each surface data_at position/value...
  for (unsigned int s=0; s<sources.size(); s++) {
    Point pt=sources[s];
    double val=vals[s];

    // find the tet nodes (nbrs) that it's closest to
    TetVolMesh::Cell::index_type tvm_cidx;
    if (tvm->locate(tvm_cidx, pt)) {
      tvm->get_nodes(nbrs, tvm_cidx);
    } else if (outside) {
      tvm->locate(nbrs[0], pt);
      nbrs.resize(1);
    } else continue;
  
    // for each nbr, see if this node is the closest of the bc nodes checked
    //   so far -- if so, store it
    unsigned int i;
    double dmin=-1;
    TetVolMesh::Node::index_type nmin;

    for (i=0; i<nbrs.size(); i++) {
      Point nbr_pt;
      TetVolMesh::Node::index_type nidx = nbrs[i];
      tvm->get_center(nbr_pt, nidx);
      double d = (pt - nbr_pt).length();
      if (i==0 || d<dmin) {
	nmin=nbrs[i];
	dmin=d;
      }
    }
    if (dmin != -1 && !have_some[nmin]) {
      pair<double, double> p(dmin, val);
      have_some[nmin]=1;
      bc_tet_nodes.add(nmin);
      closest[nmin].push_back(p);
    }
  }

  for (int i=0; i<bc_tet_nodes.size(); i++) {
    double val=0;
    int nsrcs=closest[bc_tet_nodes[i]].size();
    for (int j=0; j<nsrcs; j++)
      val+=closest[bc_tet_nodes[i]][j].second/nsrcs;
//    dirichlet.push_back(pair<int, double>(bc_tet_nodes[i],
//					  closest[bc_tet_nodes[i]].second));
//    }
    dirichlet.push_back(pair<int, double>(bc_tet_nodes[i], val));
  }
  imeshH->set_property("dirichlet", dirichlet, false);
  imeshH->set_property("conductivity_table", conds, false);
  omesh->send(imeshH);
}
} // End namespace BioPSE

namespace SCIRun {
CompileInfoHandle
InsertVoltageSourceGetPtBase::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("InsertVoltageSourceGetPt");
  static const string base_class_name("InsertVoltageSourceGetPtBase");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}

CompileInfoHandle
InsertVoltageSourceGetPtsAndValsBase::get_compile_info(const TypeDescription *field_td, const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("InsertVoltageSourceGetPtsAndVals");
  static const string base_class_name("InsertVoltageSourceGetPtsAndValsBase");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name() + ", " + loc_td->get_name());
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}
} // End namespace SCIRun
