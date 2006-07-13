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
 *  IsoRefine.cc:  Clip out parts of a field.
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
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/IsoRefine.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

int IsoRefineAlgo::hex_reorder_table[14][8] = {
  {0, 1, 2, 3, 4, 5, 6, 7},
  {1, 2, 3, 0, 5, 6, 7, 4},
  {2, 3, 0, 1, 6, 7, 4, 5},
  {3, 0, 1, 2, 7, 4, 5, 6},

  {4, 7, 6, 5, 0, 3, 2, 1},
  {5, 4, 7, 6, 1, 0, 3, 2},
  {6, 5, 4, 7, 2, 1, 0, 3},
  {7, 6, 5, 4, 3, 2, 1, 0},

  {4, 0, 3, 7, 5, 1, 2, 6},
  {1, 5, 6, 2, 0, 4, 7, 3},
  {6, 2, 1, 5, 7, 3, 0, 4},
  {3, 7, 4, 0, 2, 6, 5, 1},

  {1, 0, 4, 5, 2, 3, 7, 6},
  {3, 2, 6, 7, 0, 1, 5, 4},
};


class IsoRefine : public Module
{
private:
  GuiDouble gui_isoval_;
  GuiInt    gui_lte_;
  int       last_field_generation_;
  double    last_isoval_;
  int       last_lte_;
  int       last_matrix_generation_;

public:
  IsoRefine(GuiContext* ctx);
  virtual ~IsoRefine();

  virtual void execute();
};


DECLARE_MAKER(IsoRefine)


IsoRefine::IsoRefine(GuiContext* ctx)
  : Module("IsoRefine", ctx, Filter, "FieldsCreate", "SCIRun"),
    gui_isoval_(get_ctx()->subVar("isoval"), 0.0),
    gui_lte_(get_ctx()->subVar("lte"), 1),
    last_field_generation_(0),
    last_isoval_(0),
    last_lte_(-1),
    last_matrix_generation_(0)
{
}


IsoRefine::~IsoRefine()
{
}


void
IsoRefine::execute()
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
      last_lte_ == gui_lte_.get() &&
      oport_cached("Refined") &&
      oport_cached("Mapping"))
  {
    // We're up to date, return.
    return;
  }
  last_field_generation_ = ifieldhandle->generation;
  last_isoval_ = isoval;
  last_lte_ = gui_lte_.get();

  string ext = "";
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  if (mtd->get_name().find("HexVolMesh") != string::npos ||
      mtd->get_name().find("LatVolMesh") != string::npos)
  {
    ext = "Hex";
  }
  else if (mtd->get_name().find("QuadSurfMesh") != string::npos)
  {
    ext = "Quad";
  }
  else
  {
    error("Unsupported mesh type.  This module only works on HexVolMeshes and QuadSurfMeshes.");
    return;
  }

  if (!ifieldhandle->query_scalar_interface(this).get_rep())
  {
    error("Input field must contain scalar data.");
    return;
  }
  
  // Make the field linear if it has a constant basis.  Push the
  // constant values from the elements out to the nodes.
  if (ifieldhandle->basis_order() != 1)
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci = IRMakeLinearAlgo::get_compile_info(ftd);
    Handle<IRMakeLinearAlgo> algo;
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Unable to compile IRMakeLinear algorithm.");
      return;
    }
    ifieldhandle = algo->execute(this, ifieldhandle);
  }

  if (ext == "Hex")
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci = IRMakeConvexAlgo::get_compile_info(ftd);
    Handle<IRMakeConvexAlgo> algo;
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Unable to compile IRMakeConvex algorithm.");
      return;
    }
    ifieldhandle.detach();
    algo->execute(this, ifieldhandle);
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = IsoRefineAlgo::get_compile_info(ftd, ext);
  Handle<IsoRefineAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile IsoRefine algorithm.");
    return;
  }

  MatrixHandle interp(0);
  FieldHandle ofield = algo->execute(this, ifieldhandle,
				     isoval, gui_lte_.get(),
				     interp);
  
  send_output_handle("Refined", ofield);
  send_output_handle("Mapping", interp);
}


CompileInfoHandle
IsoRefineAlgo::get_compile_info(const TypeDescription *fsrc,
                                string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("IsoRefineAlgo" + ext);
  static const string base_class_name("IsoRefineAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_mesh_include("../src/Core/Datatypes/HexVolMesh.h");
  rval->add_basis_include("../src/Core/Basis/HexTrilinearLgn.h");
  fsrc->fill_compile_info(rval);

  return rval;
}


CompileInfoHandle
IRMakeLinearAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("IRMakeLinearAlgoT");
  static const string base_class_name("IRMakeLinearAlgo");

  const string fsrcstr = fsrc->get_name();
  const string::size_type loc = fsrcstr.find("Point");
  const string fdststr = fsrcstr.substr(0, loc) +
    "Point> > ,HexTrilinearLgn<double>, vector<double> > ";

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name,
                       fsrcstr + ", " + fdststr);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_mesh_include("../src/Core/Datatypes/HexVolMesh.h");
  rval->add_basis_include("../src/Core/Basis/HexTrilinearLgn.h");
  fsrc->fill_compile_info(rval);

  return rval;
}


CompileInfoHandle
IRMakeConvexAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("IRMakeConvexAlgoT");
  static const string base_class_name("IRMakeConvexAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_mesh_include("../src/Core/Datatypes/HexVolMesh.h");
  rval->add_basis_include("../src/Core/Basis/HexTrilinearLgn.h");
  fsrc->fill_compile_info(rval);

  return rval;
}


} // End namespace SCIRun

