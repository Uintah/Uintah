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
#include <Core/Containers/Handle.h>

#include <iostream>

namespace SCIRun {

//! Module to build a surface field from a volume field.
class FieldBoundary : public Module {
public:
  FieldBoundary(GuiContext* ctx);
  virtual ~FieldBoundary();
  virtual void execute();

private:
  
  //! Input should be a volume field.
  FieldIPort*              infield_;
  int                      infield_gen_;

  //! BoundaryField field output.
  FieldOPort*              osurf_;
  
  //! BoundaryField interpolant field output.
  FieldOPort*              ointerp_;
  
  //! Handle on the generated surface.
  FieldHandle              tri_fh_;

  //! Handle on the interpolant surface.
  FieldHandle              interp_fh_;
};

DECLARE_MAKER(FieldBoundary)
FieldBoundary::FieldBoundary(GuiContext* ctx) : 
  Module("FieldBoundary", ctx, Filter, "FieldsCreate", "SCIRun"),
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
  osurf_ = (FieldOPort *)get_oport("BoundaryField");
  ointerp_ = (FieldOPort *)get_oport("Interpolant");
  FieldHandle input;
  if (!infield_) {
    error("Unable to initialize iport 'Field'.");
    return;
  }
  if (!osurf_) {
    error("Unable to initialize oport 'BoundaryField'.");
    return;
  }
  if(!ointerp_) {
    error("Unable to initialize oport 'Interpolant'.");
    return;
  }
  if (!infield_->get(input)) return;
  if (!input.get_rep()) {
    error("FieldBoundary Error: No input data.");
    return;
  } else if (infield_gen_ != input->generation) {
    infield_gen_ = input->generation;
    MeshHandle mesh = input->mesh();

    const TypeDescription *mtd = mesh->get_type_description();
    CompileInfoHandle ci = FieldBoundaryAlgo::get_compile_info(mtd);
    Handle<FieldBoundaryAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->execute(this, mesh, tri_fh_, interp_fh_);
  }
  osurf_->send(tri_fh_);
  ointerp_->send(interp_fh_);
}


bool
FieldBoundaryAlgoAux::determine_tri_order(const Point &p0,
					  const Point &p1,
					  const Point &p2,
					  const Point &inside)
{
  const Vector v1 = p1 - p0;
  const Vector v2 = p2 - p1;
  const Vector norm = Cross(v1, v2);

  const Vector tmp = inside - p0;
  const double val = Dot(norm, tmp);
  if (val > 0.0L) {
    // normal points inside, reverse the order.
    return false;
  } else {
    // normal points outside.
    return true;
  }
}

CompileInfoHandle
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


CompileInfoHandle
FieldBoundaryAlgoAux::get_compile_info(const TypeDescription *mesh_td,
				       const string &aname)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("FieldBoundaryAlgo" + aname + "T");
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


