
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
  GuiInt averaging_;
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
    averaging_(context->subVar("averaging")),
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
    CompileInfo *ci = InsertVoltageSourceGetPtBase::get_compile_info(meshtd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    InsertVoltageSourceGetPtBase *algo =
      dynamic_cast<InsertVoltageSourceGetPtBase *>(algo_handle.get_rep());
    if (algo == 0)
    {
      error("Could not get algorithm.");
      return;
    }
    Point pt = algo->execute(isourceH->mesh());
    sources.push_back(pt);
    vals.push_back(0);
  } else {
    const TypeDescription *field_td = isourceH->get_type_description();
    const TypeDescription *loc_td = isourceH->data_at_type_description();
    CompileInfo *ci = 
      InsertVoltageSourceGetPtsAndValsBase::get_compile_info(field_td, 
							     loc_td);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      error("Could not compile algorithm.");
      return;
    }
    
    InsertVoltageSourceGetPtsAndValsBase *algo = 
      dynamic_cast<InsertVoltageSourceGetPtsAndValsBase *>(algo_handle.get_rep());
    if (algo == 0) 
    {
      error("Could not get algorithm.");
      return;
    }
    algo->execute(isourceH, sources, vals);
  }

  vector<pair<int, double> > dirichlet;
  imeshH->get_property("dirichlet", dirichlet);

  vector<pair<string, Tensor> > conds;
  imeshH->get_property("conductivity_table", conds);

  // get our own local copy of the Field and mesh
  imeshH.detach();  

  int averaging = averaging_.get();
  int outside = outside_.get();

  TetVolMesh::Node::size_type tvm_nnodes;
  TetVolMesh::Cell::size_type tvm_ncells;
  tvm->size(tvm_nnodes);
  tvm->size(tvm_ncells);

  TetVolMesh::Node::array_type nbrs(4);

  Array1<Array1<pair<double, double> > > contrib(tvm_nnodes);
  Array1<TetVolMesh::Node::index_type> have_some;

  // for each surface data_at position/value...
  for (int s=0; s<(int)sources.size(); s++) {
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
  
    // for each nbr, store the value/distance in contrib[nbr]
    unsigned int i;
    for (i=0; i<nbrs.size(); i++) {
      Point nbr_pt;
      tvm->get_center(nbr_pt, nbrs[i]);
      double d = (pt - nbr_pt).length();
      if (d < 1.e-6) d=1.e-6;
      d = 1/d;
      pair<double, double> p(d, val);
      have_some.add(nbrs[i]);
      if (contrib[nbrs[i]].size() == 0) {
	if (averaging) {
	  contrib[nbrs[i]].add(p);
	} else {
	  if (contrib[nbrs[i]].size() == 0) {
	    contrib[nbrs[i]].add(p);
	  } else if (d > contrib[nbrs[i]][0].first) {
	    contrib[nbrs[i]][0] = p;
	  }
	}
      }
    }
  }

  // for each tet node that was touched, compute 1/(sum of all the distances)
  // multiply each distance by that to get weightings
  int i,j;
  for (i=0; i<have_some.size(); i++) {
    double sum_dist = 0;
    TetVolMesh::Node::index_type node = have_some[i];
    for (j=0; j<contrib[node].size(); j++) {
      sum_dist += contrib[node][j].first;
    }
    sum_dist = 1./sum_dist;
    for (j=0; j<contrib[node].size(); j++) {
      contrib[node][j].first *= sum_dist;
    }
  }

  // if no meshes changed, we'll use the same weightings -- just new values
  for (i=0; i<have_some.size(); i++) {
    TetVolMesh::Node::index_type node = have_some[i];
    double avg_val = 0;
    for (j=0; j<contrib[node].size(); j++) {
      avg_val += contrib[node][j].second * contrib[node][j].first;
    }
    dirichlet.push_back(pair<int, double>(node, avg_val));
  }
  imeshH->set_property("dirichlet", dirichlet, false);
  imeshH->set_property("conductivity_table", conds, false);
  omesh->send(imeshH);
}
} // End namespace BioPSE

namespace SCIRun {
CompileInfo *
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

CompileInfo *
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
