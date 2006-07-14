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
 *  InsertHexSheet.cc:  Remove a layer of hexes.
 *
 *  Written by:
 *   Jason Shepherd
 *   Department of Computer Science
 *   University of Utah
 *   May 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/ExtractHexSheet.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class ExtractHexSheet : public Module
{
private:

  GuiString  gui_edge_list_;
  int       last_field_generation_;

  vector< unsigned int > edge_ids_;

public:
  ExtractHexSheet(GuiContext* ctx);
  virtual ~ExtractHexSheet();

  virtual void execute();
};


DECLARE_MAKER(ExtractHexSheet)


ExtractHexSheet::ExtractHexSheet(GuiContext* ctx)
        : Module("ExtractHexSheet", ctx, Filter, "FieldsCreate", "SCIRun"),  
          gui_edge_list_(ctx->subVar("edge-list"), "No values present."),
          last_field_generation_(0)
{
}


ExtractHexSheet::~ExtractHexSheet()
{
}


void
ExtractHexSheet::execute()
{
  // Get input fields.
  FieldHandle hexfieldhandle;
  if (!get_input_handle("HexField", hexfieldhandle)) return;

  bool changed = false;

  vector<unsigned int> edgeids(0);
  istringstream vlist(gui_edge_list_.get());
  unsigned int val;
  while(!vlist.eof()) 
  {
    vlist >> val;
    if (vlist.fail()) 
    {
      if (!vlist.eof()) 
      {
        vlist.clear();
        warning("List of Edge Ids was bad at character " +
                to_string((int)(vlist.tellg())) +
                "('" + ((char)(vlist.peek())) + "').");
      }
      break;
    }

    edgeids.push_back(val);
  }
  
    // See if any of the isovalues have changed.
  if( edge_ids_.size() != edgeids.size() ) 
  {
    edge_ids_.resize( edgeids.size() );
    changed = true;
  }

  for( unsigned int i=0; i<edgeids.size(); i++ )
  {
    if( edge_ids_[i] != edgeids[i] ) 
    {
      edge_ids_[i] = edgeids[i];
      changed = true;
    }
  }

  if (last_field_generation_ == hexfieldhandle->generation &&
      oport_cached( "NewHexField" )&&
      oport_cached( "ExtractedHexes" ) && !changed )
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
    error( "Only HexVolFields are currently supported in the ExtractHexSheet module.");
    return;
  }

  const TypeDescription *ftd = hexfieldhandle->get_type_description();
  CompileInfoHandle ci = ExtractHexSheetAlgo::get_compile_info(ftd, ext);
  Handle<ExtractHexSheetAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile ExtractHexSheet algorithm.");
    return;
  }

  FieldHandle keptfield, extractedfield;
  algo->execute( this, hexfieldhandle, edgeids, keptfield, extractedfield );
  
  send_output_handle("NewHexField", keptfield);
  send_output_handle("ExtractedHexes", extractedfield);
}

CompileInfoHandle
ExtractHexSheetAlgo::get_compile_info(const TypeDescription *fsrc,
			      string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("ExtractHexSheetAlgo" + ext);
  static const string base_class_name("ExtractHexSheetAlgo");

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

