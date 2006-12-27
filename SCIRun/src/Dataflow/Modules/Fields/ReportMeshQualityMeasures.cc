/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  ReportMeshQualityMeasures.cc:  Smooth a given mesh.
 *
 *  Written by:
 *   Jason Shepherd
 *   Department of Computer Science
 *   University of Utah
 *   January 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/ReportMeshQualityMeasures.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class ReportMeshQualityMeasures : public Module
{
private:
  int       last_field_generation_;

public:
  ReportMeshQualityMeasures(GuiContext* ctx);
  virtual ~ReportMeshQualityMeasures();

  virtual void execute();
};


DECLARE_MAKER(ReportMeshQualityMeasures)


ReportMeshQualityMeasures::ReportMeshQualityMeasures(GuiContext* ctx)
        : Module("ReportMeshQualityMeasures", ctx, Filter, "ChangeFieldData", "SCIRun"),
          last_field_generation_(0)
{
}


ReportMeshQualityMeasures::~ReportMeshQualityMeasures()
{
}


void
ReportMeshQualityMeasures::execute()
{
  // Get input field.
  FieldHandle ifieldhandle;
  if (!get_input_handle("Input", ifieldhandle)) return;

  if (last_field_generation_ == ifieldhandle->generation &&
      oport_cached("Checked") )
  {
    // We're up to date, return.
    return;
  }
  last_field_generation_ = ifieldhandle->generation;

  string ext = "";
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  if (mtd->get_name().find("TetVolMesh") != string::npos)
  {
    ext = "Tet";  
  }
  else if (mtd->get_name().find("TriSurfMesh") != string::npos)
  {
    ext = "Tri"; 
  }
  else if (mtd->get_name().find("HexVolMesh") != string::npos)
  {
    ext = "Hex"; 
  }
  else if (mtd->get_name().find("QuadSurfMesh") != string::npos)
  {
    ext = "Quad"; 
  }
  else if (mtd->get_name().find("LatVolMesh") != string::npos)
  {
    error("LatVolFields are not directly supported in this module.  Please first convert it into a HexVolField by inserting an upstream SCIRun::FieldsGeometry::Unstructure module.");
    return;
  }
  else if (mtd->get_name().find("ImageMesh") != string::npos)
  {
    error("ImageFields are not currently supported in the ReportMeshQualityMeasures module.");
    return;
  }
  else
  {
    error("Unsupported mesh type.  This module only works on Tets, Hexes, Quads, and Tris.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = ReportMeshQualityMeasuresAlgo::get_compile_info(ftd, ext);
  Handle<ReportMeshQualityMeasuresAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile ReportMeshQualityMeasures algorithm.");
    return;
  }

  FieldHandle ofield = algo->execute(this, ifieldhandle);  

  send_output_handle("Checked", ofield);
}


CompileInfoHandle
ReportMeshQualityMeasuresAlgo::get_compile_info(const TypeDescription *fsrc,
			      string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("ReportMeshQualityMeasuresAlgo" + ext);
  static const string base_class_name("ReportMeshQualityMeasuresAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  fsrc->fill_compile_info(rval);

  return rval;
}


} // End namespace SCIRun

