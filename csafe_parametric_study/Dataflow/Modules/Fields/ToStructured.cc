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
 *  ToStructured: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   July 2004
 *
 *  Copyright (C) 2004 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ToStructured.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ToStructured : public Module
{
public:
  ToStructured(GuiContext* ctx);
  virtual ~ToStructured();
  virtual void execute();

private:
  int last_generation_;
  FieldHandle ofieldhandle_;
};


DECLARE_MAKER(ToStructured)
ToStructured::ToStructured(GuiContext* ctx)
  : Module("ToStructured", ctx, Filter, "FieldsGeometry", "SCIRun"),
    last_generation_(0)
{
}



ToStructured::~ToStructured()
{
}



void
ToStructured::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input Field", ifieldhandle)) return;

  if (!ofieldhandle_.get_rep() ||
      ifieldhandle->generation != last_generation_)
  {
    last_generation_ = ifieldhandle->generation;
    ofieldhandle_ = ifieldhandle;
    string dstname = "";
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    const string &mtdn = mtd->get_name();

    if (mtdn.substr(0, 10) == "LatVolMesh")
    {
      string dmeshname = "StructHexVolMesh" + mtdn.substr(10);
      dstname = ftd->get_similar_name(dmeshname, 0);
    }
    else if (mtdn.substr(0, 9) == "ImageMesh")
    {
      string dmeshname = "StructQuadSurfMesh" + mtdn.substr(9);
      dstname = ftd->get_similar_name(dmeshname, 0);
    }  
    else if (mtdn.substr(0, 12) == "ScanlineMesh")
    {
      string dmeshname = "StructCurveMesh" + mtdn.substr(12);
      dstname = ftd->get_similar_name(dmeshname, 0);
    }
    if (dstname == "" &&
        !(mtdn.substr(0, 16) == "StructHexVolMesh" ||
          mtdn.substr(0, 18) == "StructQuadSurfMesh" ||
          mtdn.substr(0, 15) == "StructCurveMesh"))
    {
      warning("Do not know how to structure a " + mtdn + ".");
    }
    else
    {
      CompileInfoHandle ci = ToStructuredAlgo::get_compile_info(ftd, dstname);
      Handle<ToStructuredAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;

      ofieldhandle_ = algo->execute(this, ifieldhandle);

      if (ofieldhandle_.get_rep())
      {
	ofieldhandle_->copy_properties(ifieldhandle.get_rep());
      }
    }
  }

  send_output_handle("Output Field", ofieldhandle_, true);
}



CompileInfoHandle
ToStructuredAlgo::get_compile_info(const TypeDescription *fsrc,
				   const string &partial_fdst)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ToStructuredAlgoT");
  static const string base_class_name("ToStructuredAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       to_filename(partial_fdst) + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + "," + partial_fdst);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/HexTrilinearLgn.h");
  rval->add_basis_include("../src/Core/Basis/QuadBilinearLgn.h");
  rval->add_basis_include("../src/Core/Basis/CrvLinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/LatVolMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/ImageMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/ScanlineMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/StructHexVolMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/StructQuadSurfMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/StructCurveMesh.h");
  fsrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
