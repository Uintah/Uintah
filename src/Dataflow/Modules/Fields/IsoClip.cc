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

#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/IsoClip.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <stack>

namespace SCIRun {

int IsoClipAlgo::permute_table[15][4] = {
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

class IsoClip : public Module
{
private:
  GuiString clipfunction_;
  int  last_input_generation_;

public:
  IsoClip(GuiContext* ctx);
  virtual ~IsoClip();

  virtual void execute();
};


DECLARE_MAKER(IsoClip)

IsoClip::IsoClip(GuiContext* ctx)
  : Module("IsoClip", ctx, Filter, "Fields", "SCIRun"),
    clipfunction_(ctx->subVar("clipfunction")),
    last_input_generation_(0)
{
}


IsoClip::~IsoClip()
{
}



void
IsoClip::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }
  if (!ifieldhandle->mesh()->is_editable())
  {
    error("Not an editable mesh type.");
    error("(Try passing Field through an Unstructure module first).");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = IsoClipAlgo::get_compile_info(ftd);
  Handle<IsoClipAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, true, this))
  {
    error("Unable to compile IsoClip algorithm.");
    return;
  }

  const bool lte = clipfunction_.get() == "lte";
  FieldHandle ofield = algo->execute(this, ifieldhandle, 0, lte);
  
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port)
  {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  ofield_port->send(ofield);
}



CompileInfoHandle
IsoClipAlgo::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("IsoClipAlgoT");
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

