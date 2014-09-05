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
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/MaskLattice.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <iostream>

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
    maskfunction_(get_ctx()->subVar("maskfunction"), "v > 0")
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
  if (ifieldhandle->basis_order() != 0)
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

  for( ;; )
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    const TypeDescription *ltd = ifieldhandle->order_type_description();
    CompileInfoHandle ci =
      MaskLatticeAlgo::get_compile_info(ftd, ltd, maskfunc, hoff);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Your function would not compile.");
      get_gui()->eval(get_id() + " compile_error "+ci->filename_);
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
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
  ofp->send_and_dereference(ofield);
}


CompileInfoHandle
MaskLatticeAlgo::get_compile_info(const TypeDescription *field_td,
				  const TypeDescription *loc_td,
				  string clipfunction,
				  int hashoffset)
{
  unsigned int hashval = Hash(clipfunction, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("MaskLatticeInstance" + to_string(hashval));
  static const string base_class_name("MaskLatticeAlgo");

  string maskedfieldname = field_td->get_name();
  const string::size_type loc1 = maskedfieldname.find("LatVolMesh");
  maskedfieldname.insert(loc1, "Masked");
  const string::size_type loc2 = maskedfieldname.find("LatVolMesh", loc1+16);
  maskedfieldname.insert(loc2, "Masked");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       maskedfieldname + ", " +
		       "Masked" + loc_td->get_name() + ", " + 
		       field_td->get_name() + ", " +
		       loc_td->get_name());

  // Code for the clip function.
  string class_declaration =
    string("template <class A, class B, class C, class D>\n") +
    "class " + template_name + " : public MaskLatticeAlgoT<A, B, C, D>\n" +
    "{\n" +
    "  virtual bool vinside_p(double x, double y, double z,\n" +
    "                         typename A::value_type v)\n" +
    "  {\n" +
    "    return " + clipfunction + ";\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(clipfunction) + "\"); }\n" +
    "};\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/HexTrilinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/MaskedLatVolMesh.h");
  rval->add_post_include(class_declaration);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

