/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  FieldBoundary.cc:  Build a surface field from a volume field
 *
 *  Written by:
 *   Martin Cole
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 1998 SCI Group
 */

#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>

#include <iostream>

namespace SCIRun {

//! Module to build a surface field from a volume field.
class FieldBoundary : public Module {
public:
  FieldBoundary(const string& id);
  virtual ~FieldBoundary();
  virtual void execute();

private:
  
  //! Input should be a volume field.
  FieldIPort*              infield_;
  int                      infield_gen_;

  //! TriSurf field output.
  FieldOPort*              osurf_;
  
  //! TriSurf interpolant field output.
  FieldOPort*              ointerp_;
  
  //! Handle on the generated surface.
  FieldHandle              tri_fh_;

  //! Handle on the interpolant surface.
  FieldHandle              interp_fh_;
};

extern "C" Module* make_FieldBoundary(const string& id)
{
  return scinew FieldBoundary(id);
}

FieldBoundary::FieldBoundary(const string& id) : 
  Module("FieldBoundary", id, Filter, "Fields", "SCIRun"),
  infield_gen_(-1),
  tri_fh_(0), interp_fh_(0)
{
}

FieldBoundary::~FieldBoundary()
{
}


void 
FieldBoundary::execute()
{
  infield_ = (FieldIPort *)get_iport("Field");
  osurf_ = (FieldOPort *)get_oport("TriSurf");
  ointerp_ = (FieldOPort *)get_oport("Interpolant");
  FieldHandle input;
  if (!infield_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!osurf_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if(!ointerp_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!infield_->get(input)) return;
  if (!input.get_rep()) {
    error("FieldBoundary Error: No input data.");
    return;
  } else if (infield_gen_ != input->generation) {
    infield_gen_ = input->generation;
    MeshHandle mesh = input->mesh();
    mesh->finish_mesh();

    const TypeDescription *mtd = mesh->get_type_description();
    CompileInfo *ci = FieldBoundaryAlgo::get_compile_info(mtd);
    DynamicAlgoHandle algo_handle;
    if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
    {
      cout << "Could not compile algorithm." << std::endl;
      return;
    }
    FieldBoundaryAlgo *algo =
      dynamic_cast<FieldBoundaryAlgo *>(algo_handle.get_rep());
    if (algo == 0)
    {
      cout << "Could not get algorithm." << std::endl;
      return;
    }
    algo->execute(mesh, tri_fh_, interp_fh_);
  }
  osurf_->send(tri_fh_);
  ointerp_->send(interp_fh_);
}


bool
FieldBoundaryAlgoAux::determine_tri_order(const Point p[3],
					  const Point &inside)
{
  const Vector v1 = p[1] - p[0];
  const Vector v2 = p[2] - p[1];
  const Vector norm = Cross(v1, v2);

  const Vector tmp = inside - p[0];
  const double val = Dot(norm, tmp);
  if (val > 0.0L) {
    // normal points inside, reverse the order.
    return false;
  } else {
    // normal points outside.
    return true;
  }
}

CompileInfo *
FieldBoundaryAlgo::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldBoundaryAlgoT");
  static const string base_class_name("FieldBoundaryAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_name(".", ".") + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}


CompileInfo *
FieldBoundaryAlgoAux::get_compile_info(const TypeDescription *mesh_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldBoundaryAlgoAuxT");
  static const string base_class_name("FieldBoundaryAlgoAux");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       mesh_td->get_name(".", ".") + "." ,
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun


