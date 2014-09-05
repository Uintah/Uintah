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
 *  InsertHexVolSheetFromTriSurf.cc:  Insert a layer of hexes.
 *
 *  Written by:
 *   Jason Shepherd
 *   Department of Computer Science
 *   University of Utah
 *   March 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/InsertHexVolSheetFromTriSurf.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class InsertHexVolSheetFromTriSurf : public Module
{
private:
  GuiString add_to_side_;
  GuiString add_layer_;
  int       last_field_generation_;

  string last_add_to_side_;
  string last_add_layer_;
  
public:
  InsertHexVolSheetFromTriSurf(GuiContext* ctx);
  virtual ~InsertHexVolSheetFromTriSurf();

  virtual void execute();
};


DECLARE_MAKER(InsertHexVolSheetFromTriSurf)


InsertHexVolSheetFromTriSurf::InsertHexVolSheetFromTriSurf(GuiContext* ctx)
        : Module("InsertHexVolSheetFromTriSurf", ctx, Filter, "NewField", "SCIRun"),
          add_to_side_(get_ctx()->subVar("side"), "side1" ),
          add_layer_(get_ctx()->subVar("addlayer"), "On" ),
          last_field_generation_(0)
{
}


InsertHexVolSheetFromTriSurf::~InsertHexVolSheetFromTriSurf()
{
}


void
InsertHexVolSheetFromTriSurf::execute()
{
  // Get input fields.
  FieldHandle hexfieldhandle;
  if (!get_input_handle("HexField", hexfieldhandle)) return;

  FieldHandle trifieldhandle;
  if (!get_input_handle("TriField", trifieldhandle)) return;

  bool changed = false;
  add_to_side_.reset();
  if( last_add_to_side_ != add_to_side_.get() )
  {
    last_add_to_side_ = add_to_side_.get();
    changed = true;
  } 
  if( last_add_layer_ != add_layer_.get() )
  {
    last_add_layer_ = add_layer_.get();
    changed = true;
  }
  
  if (last_field_generation_ == hexfieldhandle->generation &&
      last_field_generation_ == trifieldhandle->generation &&
      oport_cached( "Side1Field" )&&
      oport_cached( "Side2Field" ) && !changed )
  {
    // We're up to date, return.
    return;
  }
  last_field_generation_ = hexfieldhandle->generation;

  string ext = "";
  const TypeDescription *mtd = hexfieldhandle->mesh()->get_type_description();
  if (mtd->get_name().find("HexVolMesh") != string::npos)
  {
    ext = "Hex";
  }
  else
  {
    error( "Only HexVolFields are currently supported in the InsertHexVolSheetFromTriSurf module.");
    return;
  }

  const TypeDescription *tri_mtd = trifieldhandle->mesh()->get_type_description();
  if (tri_mtd->get_name().find("TriSurfMesh") != string::npos)
  {
      //just checking... do nothing...
  }
  else
  {
    error( "Only TriSurfFields can be input to the InsertHexVolSheetFromTriSurf module.");
    return;
  }

  const TypeDescription *ftd = hexfieldhandle->get_type_description();
  CompileInfoHandle ci = InsertHexVolSheetFromTriSurfAlgo::get_compile_info(ftd, ext);
  Handle<InsertHexVolSheetFromTriSurfAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile InsertHexVolSheetFromTriSurf algorithm.");
    return;
  }

  bool add_to_side1 = false;
  if( last_add_to_side_ == "side1" )
      add_to_side1 = true;
  
  bool add_layer = false;
  if( last_add_layer_ == "On" )
      add_layer = true;
  
  FieldHandle side1field, side2field;
  algo->execute( this, hexfieldhandle, trifieldhandle, 
                 side1field, side2field, add_to_side1, add_layer );
  
  send_output_handle("Side1Field", side1field);
  send_output_handle("Side2Field", side2field);
}

CompileInfoHandle
InsertHexVolSheetFromTriSurfAlgo::get_compile_info(const TypeDescription *fsrc,
			      string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("InsertHexVolSheetFromTriSurfAlgo" + ext);
  static const string base_class_name("InsertHexVolSheetFromTriSurfAlgo");

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

