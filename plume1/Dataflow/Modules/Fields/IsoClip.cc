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
 *  IsoClip.cc:  Clip out parts of a field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/IsoClip.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

int IsoClipAlgo::tet_permute_table[15][4] = {
  { 0, 0, 0, 0 }, // 0x0
  { 3, 0, 2, 1 }, // 0x1
  { 2, 3, 0, 1 }, // 0x2
  { 0, 1, 2, 3 }, // 0x3
  { 1, 2, 0, 3 }, // 0x4
  { 0, 2, 3, 1 }, // 0x5
  { 0, 3, 1, 2 }, // 0x6
  { 0, 1, 2, 3 }, // 0x7
  { 0, 1, 2, 3 }, // 0x8
  { 2, 1, 3, 0 }, // 0x9
  { 3, 1, 0, 2 }, // 0xa
  { 1, 2, 0, 3 }, // 0xb
  { 3, 2, 1, 0 }, // 0xc
  { 2, 3, 0, 1 }, // 0xd
  { 3, 0, 2, 1 }, // 0xe
};


int IsoClipAlgo::tri_permute_table[7][3] = {
  { 0, 0, 0 }, // 0x0
  { 2, 0, 1 }, // 0x1
  { 1, 2, 0 }, // 0x2
  { 0, 1, 2 }, // 0x3
  { 0, 1, 2 }, // 0x4
  { 1, 2, 0 }, // 0x5
  { 2, 0, 1 }, // 0x6
};

class IsoClip : public Module
{
private:
  GuiDouble gui_isoval_;
  GuiInt    gui_lte_;
  int       last_field_generation_;
  double    last_isoval_;
  int       last_lte_;
  int       last_matrix_generation_;

public:
  IsoClip(GuiContext* ctx);
  virtual ~IsoClip();

  virtual void execute();
};


DECLARE_MAKER(IsoClip)


IsoClip::IsoClip(GuiContext* ctx)
  : Module("IsoClip", ctx, Filter, "FieldsCreate", "SCIRun"),
    gui_isoval_(ctx->subVar("isoval")),
    gui_lte_(ctx->subVar("lte")),
    last_field_generation_(0),
    last_isoval_(0),
    last_lte_(-1),
    last_matrix_generation_(0)
{
}


IsoClip::~IsoClip()
{
}


void
IsoClip::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input");
  FieldHandle ifieldhandle;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  MatrixIPort *imp = (MatrixIPort *)get_iport("Optional Isovalue");
  MatrixHandle isomat;
  if (imp->get(isomat) && isomat.get_rep() &&
      isomat->nrows() > 0 && isomat->ncols() > 0 &&
      isomat->generation != last_matrix_generation_)
  {
    last_matrix_generation_ = isomat->generation;
    gui_isoval_.set(isomat->get(0, 0));
  }

  const double isoval = gui_isoval_.get();
  if (last_field_generation_ == ifieldhandle->generation &&
      last_isoval_ == isoval &&
      last_lte_ == gui_lte_.get())
  {
    // We're up to date, return.
    return;
  }
  last_field_generation_ = ifieldhandle->generation;
  last_isoval_ = isoval;
  last_lte_ = gui_lte_.get();

  string ext = "";
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  if (mtd->get_name() == "TetVolMesh")
  {
    ext = "Tet";
  }
  else if (mtd->get_name() == "TriSurfMesh")
  {
    ext = "Tri";
  }
  else if (mtd->get_name() == "HexVolMesh")
  {
    error("HexVolFields are not directly supported in this module.  Please first convert it into a TetVolField by inserting a SCIRun::FieldsGeometry::HexToTet module upstream.");
    return;
  }
  else if (mtd->get_name() == "QuadSurfMesh")
  {
    error("QuadSurfFields are not directly supported in this module.  Please first convert it into a TriSurfField by inserting a SCIRun::FieldsGeometry::QuadToTri module upstream.");
    return;
  }
  else if (mtd->get_name() == "LatVolMesh")
  {
    error("LatVolFields are not directly supported in this module.  Please first convert it into a TetVolField by inserting an upstream SCIRun::FieldsGeometry::Unstructure module, followed by a SCIRun::FieldsGeometry::HexToTet module.");
    return;
  }
  else if (mtd->get_name() == "ImageMesh")
  {
    error("ImageFields are not supported in this module.  Please first convert it into a TriSurfField by inserting an upstream SCIRun::FieldsGeometry::Unstructure module, followed by a SCIRun::FieldsGeometry::QuadToTri module.");
    return;
  }
  else
  {
    error("Unsupported mesh type.  This module only works on Tets and Tris.");
    return;
  }

  if (!ifieldhandle->query_scalar_interface(this).get_rep())
  {
    error("Input field must contain scalar data.");
    return;
  }
  
  if (ifieldhandle->basis_order() != 1)
  {
    error("Isoclipping can only be done for fields with data at nodes.  Note: you can insert a ChangeFieldDataAt module (and an ApplyInterpMatrix module) upstream to push element data to the nodes.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = IsoClipAlgo::get_compile_info(ftd, ext);
  Handle<IsoClipAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile IsoClip algorithm.");
    return;
  }

  MatrixHandle interp(0);
  FieldHandle ofield = algo->execute(this, ifieldhandle,
				     isoval, gui_lte_.get(),
				     interp);
  
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Clipped");
  ofield_port->send(ofield);

  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Mapping");
  omatrix_port->send(interp);
}



CompileInfoHandle
IsoClipAlgo::get_compile_info(const TypeDescription *fsrc,
			      string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("IsoClipAlgo" + ext);
  static const string base_class_name("IsoClipAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);

  return rval;
}


} // End namespace SCIRun

