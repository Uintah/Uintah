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
 *  ClipByFunction.cc:  Clip out parts of a field.
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
#include <Dataflow/Modules/Fields/ClipByFunction.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <stack>

namespace SCIRun {

class ClipByFunction : public Module
{
private:
  GuiString clipmode_;
  GuiString clipfunction_;
  int  last_input_generation_;

public:
  ClipByFunction(GuiContext* ctx);
  virtual ~ClipByFunction();

  virtual void execute();
};


DECLARE_MAKER(ClipByFunction)

ClipByFunction::ClipByFunction(GuiContext* ctx)
  : Module("ClipByFunction", ctx, Filter, "Fields", "SCIRun"),
    clipmode_(ctx->subVar("clipmode")),
    clipfunction_(ctx->subVar("clipfunction")),
    last_input_generation_(0)
{
}


ClipByFunction::~ClipByFunction()
{
}



void
ClipByFunction::execute()
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
  Handle<ClipByFunctionAlgo> algo;
  int hoffset = 0;
  while (1)
  {
    CompileInfoHandle ci =
      ClipByFunctionAlgo::get_compile_info(ftd, clipfunction_.get(), hoffset);
    if (!DynamicCompilation::compile(ci, algo, true, this))
    {
      DynamicLoader::scirun_loader().remove_cc(*(ci.get_rep()), cout);
      error("Your function would not compile.");
      return;
    }
    if (algo->identify() == clipfunction_.get())
    {
      break;
    }
    hoffset++;
  }

  int clipmode = 0;
  if (clipmode_.get() == "cell")
  {
    clipmode = 0;
  }
  else if (clipmode_.get() == "onenode")
  {
    clipmode = 1;
  }
  else if (clipmode_.get() == "allnodes")
  {
    clipmode = -1;
  }

  FieldHandle ofield =
    algo->execute(this, ifieldhandle, clipmode);
  
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port)
  {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  ofield_port->send(ofield);
}



CompileInfoHandle
ClipByFunctionAlgo::get_compile_info(const TypeDescription *fsrc,
				     string clipfunction,
				     int hashoffset)
{
  hash<const char *> H;
  unsigned int hashval = H(clipfunction.c_str()) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("ClipByFunctionInstance" + to_string(hashval));
  static const string base_class_name("ClipByFunctionAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "template <class FIELD>\n" +
    "class " + template_name + " : public ClipByFunctionAlgoT<FIELD>\n" +
    "{\n" +
    "  virtual bool vinside_p(double x, double y, double z,\n" +
    "                         typename FIELD::value_type v)\n" +
    "  {\n" +
    "    return " + clipfunction + ";\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + clipfunction + "\"); }\n" +
    "};\n//";

  rval->add_include(include_path + class_declaration);
  fsrc->fill_compile_info(rval);

  return rval;
}


} // End namespace SCIRun

