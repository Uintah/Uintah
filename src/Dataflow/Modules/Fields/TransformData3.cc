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
 *  TransformData3: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/TransformData3.h>
#include <Dataflow/Modules/Fields/FieldInfo.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>

#include <iostream>

namespace SCIRun {

class TransformData3 : public Module
{
private:
  GuiString function_;
  GuiString outputdatatype_;

public:
  TransformData3(GuiContext* ctx);
  virtual ~TransformData3();
  virtual void execute();
};


DECLARE_MAKER(TransformData3)


TransformData3::TransformData3(GuiContext* ctx)
  : Module("TransformData3", ctx, Filter,"FieldsData", "SCIRun"),
    function_(ctx->subVar("function")),
    outputdatatype_(ctx->subVar("outputdatatype"))
{
}


TransformData3::~TransformData3()
{
}


void
TransformData3::execute()
{
  // Get input field.
  FieldIPort *ifp0 = (FieldIPort *)get_iport("Input Field 0");
  FieldHandle ifieldhandle0;
  if (!ifp0) {
    error("Unable to initialize iport 'Input Field 0'.");
    return;
  }
  if (!(ifp0->get(ifieldhandle0) && ifieldhandle0.get_rep()))
  {
    error("Input field 0 is empty.");
    return;
  }


  FieldIPort *ifp1 = (FieldIPort *)get_iport("Input Field 1");
  FieldHandle ifieldhandle1;
  if (!ifp1) {
    error("Unable to initialize iport 'Input Field 1'.");
    return;
  }
  if (!(ifp1->get(ifieldhandle1) && ifieldhandle1.get_rep()))
  {
    error("Input field 1 is empty.");
    return;
  }


  FieldIPort *ifp2 = (FieldIPort *)get_iport("Input Field 2");
  FieldHandle ifieldhandle2;
  if (!ifp2) {
    error("Unable to initialize iport 'Input Field 2'.");
    return;
  }
  if (!(ifp2->get(ifieldhandle2) && ifieldhandle2.get_rep()))
  {
    error("Input field 2 is empty.");
    return;
  }

  if (ifieldhandle0->mesh().get_rep() != ifieldhandle1->mesh().get_rep() ||
      ifieldhandle0->mesh().get_rep() != ifieldhandle2->mesh().get_rep())
  {
    // If not the same mesh make sure they are the same type.
    if( ifieldhandle0->get_type_description(0)->get_name() !=
	ifieldhandle1->get_type_description(0)->get_name() ||
	ifieldhandle0->get_type_description(1)->get_name() !=
	ifieldhandle1->get_type_description(1)->get_name() ||
	
	ifieldhandle0->get_type_description(0)->get_name() !=
	ifieldhandle2->get_type_description(0)->get_name() ||
	ifieldhandle0->get_type_description(1)->get_name() !=
	ifieldhandle2->get_type_description(1)->get_name() ) {
      error("The input fields must have the same mesh type.");
      return;
    } else {

      // Do this last, sometimes takes a while.
      const TypeDescription *meshtd0 = ifieldhandle0->mesh()->get_type_description();

      CompileInfoHandle ci = FieldInfoAlgoCount::get_compile_info(meshtd0);
      Handle<FieldInfoAlgoCount> algo;
      if (!module_dynamic_compile(ci, algo)) return;

      //string num_nodes, num_elems;
      //int num_nodes, num_elems;
      const string num_nodes0 = algo->execute_node(ifieldhandle0->mesh());
      const string num_elems0 = algo->execute_elem(ifieldhandle0->mesh());

      const string num_nodes1 = algo->execute_node(ifieldhandle1->mesh());
      const string num_elems1 = algo->execute_elem(ifieldhandle1->mesh());

      const string num_nodes2 = algo->execute_node(ifieldhandle2->mesh());
      const string num_elems2 = algo->execute_elem(ifieldhandle2->mesh());

      if( num_nodes0 != num_nodes1 || num_nodes0 != num_nodes2 ||
	  num_elems0 != num_elems1 || num_elems0 != num_elems2 ) {
	error("The input meshes must have the same number of nodes and elements.");
	return;
      } else {
	warning("The input fields do not have the same mesh,");
	warning("but appear to be the same otherwise.");
      }
    }
  }

  if (ifieldhandle0->basis_order() != ifieldhandle1->basis_order() ||
      ifieldhandle0->basis_order() != ifieldhandle2->basis_order())
  {
    error("The Input Fields must share the same data location.");
    return;
  }

  string outputdatatype = outputdatatype_.get();
  if (outputdatatype == "input 0")
  {
    outputdatatype = ifieldhandle0->get_type_description(1)->get_name();
  }
  else if (outputdatatype == "input 1")
  {
    outputdatatype = ifieldhandle1->get_type_description(1)->get_name();
  }
  else if (outputdatatype == "input 2")
  {
    outputdatatype = ifieldhandle2->get_type_description(1)->get_name();
  }


  const TypeDescription *ftd0 = ifieldhandle0->get_type_description();
  const TypeDescription *ftd1 = ifieldhandle1->get_type_description();
  const TypeDescription *ftd2 = ifieldhandle2->get_type_description();
  const TypeDescription *ltd = ifieldhandle0->order_type_description();
  const string oftn = ifieldhandle0->get_type_description(0)->get_name() +
    "<" + outputdatatype + "> ";
  int hoffset = 0;
  Handle<TransformData3Algo> algo;

  // remove trailing white-space from the function string
  string func=function_.get();
  while (func.size() && isspace(func[func.size()-1]))
    func.resize(func.size()-1);

  while (1)
  {
    CompileInfoHandle ci =
      TransformData3Algo::get_compile_info(ftd0, ftd1, ftd2, oftn, ltd,
					   func, hoffset);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Your function would not compile.");
      gui->eval(id + " compile_error "+ci->filename_);
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return;
    }
    if (algo->identify() == func)
    {
      break;
    }
    hoffset++;
  }

  FieldHandle ofieldhandle = algo->execute(ifieldhandle0,
					   ifieldhandle1,
					   ifieldhandle2);

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
TransformData3Algo::get_compile_info(const TypeDescription *field0_td,
				     const TypeDescription *field1_td,
				     const TypeDescription *field2_td,
				     string ofieldtypename,
				     const TypeDescription *loc_td,
				     string function,
				     int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("TransformData3Instance" + to_string(hashval));
  static const string base_class_name("TransformData3Algo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       field0_td->get_filename() + "." +
		       field1_td->get_filename() + "." +
		       field2_td->get_filename() + "." +
		       to_filename(ofieldtypename) + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       field0_td->get_name() + ", " +
                       field1_td->get_name() + ", " +
                       field2_td->get_name() + ", " +
		       ofieldtypename + ", " +
		       loc_td->get_name());

  // Code for the function.
  string class_declaration =
    string("") + 
    "template <class IFIELD0, class IFIELD1, class IFIELD2, class OFIELD, class LOC>\n" +
    "class " + template_name + " : public TransformData3AlgoT<IFIELD0, IFIELD1, IFIELD2, OFIELD, LOC>\n" +
    "{\n" +
    "  virtual void function(typename OFIELD::value_type &result,\n" +
    "                        double x, double y, double z,\n" +
    "                        const typename IFIELD0::value_type &v0,\n" +
    "                        const typename IFIELD1::value_type &v1,\n" +
    "                        const typename IFIELD2::value_type &v2)\n" +
    "  {\n" +
    "    " + function + "\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(function) + "\"); }\n" +
    "};\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  field0_td->fill_compile_info(rval);
  field1_td->fill_compile_info(rval);
  field2_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
