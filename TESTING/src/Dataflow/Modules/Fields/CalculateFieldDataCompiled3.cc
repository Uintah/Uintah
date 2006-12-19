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
 *  CalculateFieldDataCompiled3: Unary field data operations
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
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Fields/CalculateFieldDataCompiled3.h>
#include <Core/Algorithms/Fields/FieldsAlgo.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>

#include <iostream>

namespace SCIRun {

class CalculateFieldDataCompiled3 : public Module
{
private:
private:
  GuiString gFunction_;
  GuiString gOutputDataType_;

  string function_;
  string outputDataType_;

  FieldHandle fHandle_;

  int fGeneration0_;
  int fGeneration1_;
  int fGeneration2_;

  bool error_;

public:
  CalculateFieldDataCompiled3(GuiContext* ctx);
  virtual ~CalculateFieldDataCompiled3();
  virtual void execute();
  virtual void presave();
};


DECLARE_MAKER(CalculateFieldDataCompiled3)


CalculateFieldDataCompiled3::CalculateFieldDataCompiled3(GuiContext* ctx)
  : Module("CalculateFieldDataCompiled3", ctx, Filter,"ChangeFieldData", "SCIRun"),
    gFunction_(get_ctx()->subVar("function"), "result = v0 + v1 + v2;"),
    gOutputDataType_(get_ctx()->subVar("outputdatatype"), "input 0"),
    fGeneration0_(-1),
    fGeneration1_(-1),
    fGeneration2_(-1),
    error_(0)
{
}


CalculateFieldDataCompiled3::~CalculateFieldDataCompiled3()
{
}


void
CalculateFieldDataCompiled3::execute()
{
  // Get input field.
  FieldHandle fHandle0;
  if (!get_input_handle("Input Field 0", fHandle0)) return;

  if (fHandle0->basis_order() == -1)
  {
    error("Field 0 contains no data to transform.");
    return;
  }

  FieldHandle fHandle1;
  if (!get_input_handle("Input Field 1", fHandle1)) return;

  if (fHandle1->basis_order() == -1)
  {
    error("Field 1 contains no data to transform.");
    return;
  }

  FieldHandle fHandle2;
  if (!get_input_handle("Input Field 2", fHandle2)) return;

  if (fHandle2->basis_order() == -1)
  {
    error("Field 2 contains no data to transform.");
    return;
  }

  bool update = false;

  // Check to see if the source field has changed.
  if( fGeneration0_ != fHandle0->generation ) {
    fGeneration0_ = fHandle0->generation;
    update = true;
  }

  // Check to see if the source field has changed.
  if( fGeneration1_ != fHandle1->generation ) {
    fGeneration1_ = fHandle1->generation;
    update = true;
  }

  // Check to see if the source field has changed.
  if( fGeneration2_ != fHandle2->generation ) {
    fGeneration2_ = fHandle2->generation;
    update = true;
  }

  string outputDataType = gOutputDataType_.get();
  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before get.
  string function = gFunction_.get();

  if( outputDataType_ != outputDataType ||
      function_       != function ) {
    update = true;
    
    outputDataType_ = outputDataType;
    function_       = function;
  }

  if( !fHandle_.get_rep() ||
      update ||
      error_ )
  {
    error_ = false;

    // remove trailing white-space from the function string
    while (function.size() && isspace(function[function.size()-1]))
      function.resize(function.size()-1);

    if (fHandle0->basis_order() != fHandle1->basis_order() ||
	fHandle0->basis_order() != fHandle2->basis_order()) {
	error("The Input Fields must share the same data location.");
	error_ = true;
	return;
      }
    
    if (fHandle0->mesh().get_rep() != fHandle1->mesh().get_rep() ||
	fHandle0->mesh().get_rep() != fHandle2->mesh().get_rep()) {
      // If not the same mesh make sure they are the same type.
      if( fHandle0->get_type_description(Field::MESH_TD_E)->get_name() !=
	  fHandle1->get_type_description(Field::MESH_TD_E)->get_name() ||	
	  fHandle0->get_type_description(Field::MESH_TD_E)->get_name() !=
	  fHandle2->get_type_description(Field::MESH_TD_E)->get_name() ||
	  fHandle1->get_type_description(Field::MESH_TD_E)->get_name() !=
	  fHandle2->get_type_description(Field::MESH_TD_E)->get_name() ) {
	error("The input fields must have the same mesh type.");
	error_ = true;
	return;
      }

      // Do this last, sometimes takes a while.
      // Code to replace the old FieldCountAlgorithm
      SCIRunAlgo::FieldsAlgo algo(this);
      int num_nodes0, num_nodes1, num_nodes2;
      int num_elems0, num_elems1, num_elems2;
      if (!(algo.GetFieldInfo(fHandle0,num_nodes0,num_elems0))) return;
      if (!(algo.GetFieldInfo(fHandle1,num_nodes1,num_elems1))) return;
      if (!(algo.GetFieldInfo(fHandle2,num_nodes2,num_elems2))) return;

      if( num_nodes0 != num_nodes1 || num_nodes0 != num_nodes2 ||
	  num_elems0 != num_elems1 || num_elems0 != num_elems2 ) {
	error("The input meshes must have the same number of nodes and elements.");
	error_ = true;
	return;
      } else {
	warning("The input fields do not have the same mesh,");
	warning("but appear to be the same otherwise.");
      }
    }


    if (outputDataType == "input 0") {
      TypeDescription::td_vec *tdv = 
	fHandle0->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      outputDataType = (*tdv)[0]->get_name();
    }
    else if (outputDataType == "input 1") {
      TypeDescription::td_vec *tdv = 
	fHandle1->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      outputDataType = (*tdv)[0]->get_name();
    } 
    else if (outputDataType == "input 2") {
      TypeDescription::td_vec *tdv = 
	fHandle2->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      outputDataType = (*tdv)[0]->get_name();
    }

    const TypeDescription *ftd0 = fHandle0->get_type_description();
    const TypeDescription *ftd1 = fHandle1->get_type_description();
    const TypeDescription *ftd2 = fHandle2->get_type_description();
    const TypeDescription *ltd = fHandle0->order_type_description();
    const string oftn = 
      fHandle0->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<" +
      fHandle0->get_type_description(Field::MESH_TD_E)->get_name() + ", " +
      fHandle0->get_type_description(Field::BASIS_TD_E)->get_similar_name(outputDataType, 
							  0, "<", " >, ") +
      fHandle0->get_type_description(Field::FDATA_TD_E)->get_similar_name(outputDataType,
							  0, "<", " >") + " >";
    int hoffset = 0;
    Handle<CalculateFieldDataCompiled3Algo> algo;

    while (1) {
      CompileInfoHandle ci =
	CalculateFieldDataCompiled3Algo::get_compile_info(ftd0, ftd1, ftd2, oftn, ltd,
					     function, hoffset);
      if (!DynamicCompilation::compile(ci, algo, false, this)) {
	error("Your function would not compile.");
	get_gui()->eval(get_id() + " compile_error "+ci->filename_);
	DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	return;
      }
      if (algo->identify() == function)
	break;

      hoffset++;
    }

    fHandle_ = algo->execute(fHandle0, fHandle1, fHandle2);
  }

  send_output_handle("Output Field", fHandle_, true);
}


void
CalculateFieldDataCompiled3::presave()
{
  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before saving.
}


CompileInfoHandle
CalculateFieldDataCompiled3Algo::get_compile_info(const TypeDescription *field0_td,
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
  const string template_name("CalculateFieldDataCompiled3Instance" + to_string(hashval));
  static const string base_class_name("CalculateFieldDataCompiled3Algo");

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
    "class " + template_name + " : public CalculateFieldDataCompiled3AlgoT<IFIELD0, IFIELD1, IFIELD2, OFIELD, LOC>\n" +
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
  rval->add_data_include("../src/Core/Geometry/Vector.h");
  rval->add_data_include("../src/Core/Geometry/Tensor.h");
  return rval;
}


} // End namespace SCIRun
