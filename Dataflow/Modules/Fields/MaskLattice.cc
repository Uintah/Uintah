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
 *  MaskLattice.cc:  Make an ImageField that fits the source field.
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/MaskLattice.h>
#include <Core/Util/DynamicCompilation.h>

#include <iostream>
#include <sci_hash_map.h>

namespace SCIRun {

class MaskLattice : public Module
{
public:
  MaskLattice(GuiContext* ctx);
  virtual ~MaskLattice();

  virtual void execute();

private:
  GuiString maskfunction_;
};


DECLARE_MAKER(MaskLattice)



MaskLattice::MaskLattice(GuiContext* ctx)
  : Module("MaskLattice", ctx, Filter, "FieldsData", "SCIRun"),
    maskfunction_(ctx->subVar("maskfunction"))
{
}



MaskLattice::~MaskLattice()
{
}

void
MaskLattice::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }

  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("MaskLattice Module requires input.");
    return;
  }

  if (!ifieldhandle->query_scalar_interface(this).get_rep())
  {
    error("This module only works on fields containing scalar data.");
    return;
  }
  if (ifieldhandle->data_at() != Field::CELL)
  {
    error("This module currently only works on fields containing data at cells.");
    return;
  }
  
  Handle<MaskLatticeAlgo> algo;
  int hoff = 0;

  // remove trailing white-space from the function string
  string maskfunc=maskfunction_.get();
  while (maskfunc.size() && isspace(maskfunc[maskfunc.size()-1]))
    maskfunc.resize(maskfunc.size()-1);

  while (1)
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    const TypeDescription *ltd = ifieldhandle->data_at_type_description();
    CompileInfoHandle ci =
      MaskLatticeAlgo::get_compile_info(ftd, ltd, maskfunc, hoff);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      error("Your function would not compile.");
      return;
    }
    if (algo->identify() == maskfunc)
    {
      break;
    }
    hoff++;
  }
  FieldHandle ofield(algo->execute(ifieldhandle));

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Masked Field");
  if (!ofp) 
  {
    error("Unable to initialize oport 'Output Sample Field'.");
    return;
  }
  ofp->send(ofield);
}


CompileInfoHandle
MaskLatticeAlgo::get_compile_info(const TypeDescription *field_td,
				  const TypeDescription *loc_td,
				  string clipfunction,
				  int hashoffset)
{
  hash<const char *> H;
  unsigned int hashval = H(clipfunction.c_str()) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("MaskLatticeInstance" + to_string(hashval));
  static const string base_class_name("MaskLatticeAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       "Masked" + field_td->get_name() + ", " +
		       "Masked" + loc_td->get_name() + ", " + 
		       field_td->get_name() + ", " +
		       loc_td->get_name());

  // Code for the clip function.
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "template <class A, class B, class C, class D>\n" +
    "class " + template_name + " : public MaskLatticeAlgoT<A, B, C, D>\n" +
    "{\n" +
    "  virtual bool vinside_p(double x, double y, double z,\n" +
    "                         typename A::value_type v)\n" +
    "  {\n" +
    "    return " + clipfunction + ";\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + clipfunction + "\"); }\n" +
    "};\n//";

  // Add in the include path to compile this obj
  rval->add_include(include_path + class_declaration);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

