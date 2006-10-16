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
 *  SetupFEMatrix.cc:  Setups the global finite element matrix
 *
 *  Written by:
 *   F. B. Sachse
 *   CVRTI
 *   University of Utah
 *   Nov 2005
 *
 *  Generalized version of code from  
 *   Ruth Nicholson Klepfer, Department of Bioengineering
 *   University of Utah, Oct 1994
 *   Alexei Samsonov, Department of Computer Science
 *   University of Utah, Mar 2001    
 *   Sascha Moehrs, SCI , University of Utah, January 2003 (Hex)
 *   Lorena Kreda, Northeastern University, November 2003 (Tri)
 */

#include <Packages/BioPSE/Dataflow/Modules/Forward/SetupFEMatrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>

namespace BioPSE {

using namespace SCIRun;


class SetupFEMatrix : public Module
{
  GuiInt uiUseCond_;
  GuiInt uiUseBasis_;
  GuiString uiNProcessors_;
  
public:
  //! Constructor/Destructor
  SetupFEMatrix(GuiContext *context);
  virtual ~SetupFEMatrix();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(SetupFEMatrix)


SetupFEMatrix::SetupFEMatrix(GuiContext *context) : 
  Module("SetupFEMatrix", context, Filter, "Forward", "BioPSE"), 
  uiUseCond_(context->subVar("UseCondTCL")),
  uiUseBasis_(context->subVar("UseBasisTCL")),
  uiNProcessors_(context->subVar("NProcessorsTCL"))
{}


SetupFEMatrix::~SetupFEMatrix()
{
}


void
SetupFEMatrix::execute()
{
  SetupFEMatrixParam SFP;

  if (!get_input_handle("Mesh", SFP.fieldH_)) return;

  const TypeDescription *ftd = SFP.fieldH_->get_type_description(Field::FIELD_NAME_ONLY_E);
  const TypeDescription *mtd = SFP.fieldH_->get_type_description(Field::MESH_TD_E);
  const TypeDescription *btd = SFP.fieldH_->get_type_description(Field::BASIS_TD_E);
  const TypeDescription *dtd = SFP.fieldH_->get_type_description(Field::FDATA_TD_E);
  const TypeDescription::td_vec *htdv = dtd->get_sub_type();

  const string hfvaltype = (*htdv)[0]->get_name();
  if (hfvaltype != "int" && hfvaltype != "Tensor") {
    error("Input Field is not of type 'int' or 'Tensor'.");
    return; 
  }

  SFP.gen_=SFP.fieldH_->generation;
  SFP.UseCond_=uiUseCond_.get();
  SFP.UseBasis_=uiUseBasis_.get();
  SFP.nprocessors_=atoi(uiNProcessors_.get().c_str());

  CompileInfoHandle ci =
    SetupFEMatrixAlgo::get_compile_info(ftd, mtd, btd, dtd);
  Handle<SetupFEMatrixAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) 
    return;
  
  MatrixHandle hGblMtrx = algo->execute(SFP);
  
  // Send the data downstream
  send_output_handle("Stiffness Matrix", hGblMtrx);
}

CompileInfoHandle
SetupFEMatrixAlgo::get_compile_info(const TypeDescription *ftd,
			       const TypeDescription *mtd,
			       const TypeDescription *btd,
			       const TypeDescription *dtd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SetupFEMatrixAlgoT");
  static const string base_class_name("SetupFEMatrixAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       mtd->get_filename() + "."+
		       btd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + "<" + mtd->get_name() + ", " +
		       btd->get_name() + ", " + dtd->get_name() + "> "  );
  
  // Add in the include path to compile this obj

  rval->add_include(include_path);
  rval->add_namespace("BioPSE");

  ftd->fill_compile_info(rval);
  mtd->fill_compile_info(rval);
  btd->fill_compile_info(rval);
  dtd->fill_compile_info(rval);

  return rval;
}

} // End namespace BioPSE



