/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <Dataflow/Modules/Fields/ApplyMappingMatrix.h>

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
  MatrixOPort*              ointerp_;
  
  //! Handle on the generated surface.
  FieldHandle              tri_fh_;

  //! Handle on the interpolant surface.
  MatrixHandle              interp_mh_;
};

DECLARE_MAKER(FieldBoundary)
FieldBoundary::FieldBoundary(GuiContext* ctx) : 
  Module("FieldBoundary", ctx, Filter, "FieldsCreate", "SCIRun"),
  infield_gen_(-1),
  tri_fh_(0), interp_mh_(0)
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
  ointerp_ = (MatrixOPort *)get_oport("Mapping");

  FieldHandle input;
  if (!(infield_->get(input) && input.get_rep()))
  {
    error("No input field data.");
    return;
  }
  else if (infield_gen_ != input->generation)
  {
    infield_gen_ = input->generation;
    MeshHandle mesh = input->mesh();

    const TypeDescription *mtd = mesh->get_type_description();
    CompileInfoHandle ci = FieldBoundaryAlgo::get_compile_info(mtd);
    Handle<FieldBoundaryAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->execute(this, mesh, tri_fh_, interp_mh_, input->basis_order());

    // Automatically apply the interpolant matrix to the output field.
    if (tri_fh_.get_rep() && interp_mh_.get_rep())
    {
      string actype = input->get_type_description(1)->get_name();
      if (input->query_scalar_interface(this) != NULL) { actype = "double"; }
      const TypeDescription *iftd = input->get_type_description();
      const TypeDescription *iltd = input->order_type_description();
      const TypeDescription *oftd = tri_fh_->get_type_description();
      const TypeDescription *oltd = tri_fh_->order_type_description();
      CompileInfoHandle ci =
	ApplyMappingMatrixAlgo::get_compile_info(iftd, iltd,
                                                 oftd, oltd,
                                                 actype, false);
      Handle<ApplyMappingMatrixAlgo> algo;
      if (module_dynamic_compile(ci, algo))
      {
	tri_fh_ = algo->execute(input, tri_fh_->mesh(),
				interp_mh_, tri_fh_->basis_order());
      }
    }
  }

  if (!(interp_mh_.get_rep()))
  {
    warning("Interpolation for these particular field types and/or locations is not supported.");
    warning("Use the DirectInterpolate module if interpolation is required.");
  }

  osurf_->send(tri_fh_);
  ointerp_->send(interp_mh_);
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


