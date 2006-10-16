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
 *  ApplyFEMVoltageSource.cc:  Builds the RHS of the FE matrix for voltage sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *  Modified by:
 *   Alexei Samsonov, March 2001
 *   Frank B. Sachse, February 2006
 *  
 */

#include <Packages/BioPSE/Dataflow/Modules/Forward/ApplyFEMVoltageSource.h>


namespace BioPSE {

class ApplyFEMVoltageSource : public Module
{
  GuiString bcFlag_; // "none", "GroundZero", or "DirSub"

public:
  ApplyFEMVoltageSource(GuiContext *context);
  virtual ~ApplyFEMVoltageSource();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(ApplyFEMVoltageSource)


ApplyFEMVoltageSource::ApplyFEMVoltageSource(GuiContext *context) : 
  Module("ApplyFEMVoltageSource", context, Filter, "Forward", "BioPSE"),
      bcFlag_(context->subVar("bcFlag"))
{
  cerr << "ApplyFEMVoltageSource" << endl;
}


ApplyFEMVoltageSource::~ApplyFEMVoltageSource()
{
}


void
ApplyFEMVoltageSource::execute()
{
  //! Obtaining handles to computation objects
  FieldHandle hField;
  if (!get_input_handle("Mesh", hField)) return;
  
  vector<pair<int, double> > dirBC;
  if (bcFlag_.get() == "DirSub") 
    if (!hField->get_property("dirichlet", dirBC))
      warning("The input field doesn't contain Dirichlet boundary conditions.");
  MatrixHandle hMatIn;
  if (!get_input_handle("Stiffness Matrix", hMatIn)) return;

  SparseRowMatrix *matIn;
  if (!(matIn = dynamic_cast<SparseRowMatrix*>(hMatIn.get_rep()))) {
    error("Input stiffness matrix wasn't sparse.");
    return;
  }
  if (matIn->nrows() != matIn->ncols()) {
    error("Input stiffness matrix wasn't square.");
    return;
  }
  
  SparseRowMatrix *mat = matIn->clone();
  
  unsigned int nsize=matIn->ncols();
  ColumnMatrix* rhs = scinew ColumnMatrix(nsize);
  
  MatrixHandle  hRhsIn;
  ColumnMatrix* rhsIn = NULL;
  
  // -- if the user passed in a vector the right size, copy it into ours 
  MatrixIPort *iportRhs_ = (MatrixIPort *)get_iport("RHS");
  if (iportRhs_->get(hRhsIn) && 
      (rhsIn=dynamic_cast<ColumnMatrix*>(hRhsIn.get_rep())) && 
      ((unsigned int)(rhsIn->nrows()) == nsize))
  {
    string units;
    if (rhsIn->get_property("units", units))
      rhs->set_property("units", units, false);
    
    for (unsigned int i=0; i < nsize; i++) 
      (*rhs)[i]=(*rhsIn)[i];
  }
  else{
    rhs->set_property("units", string("volts"), false);
    rhs->zero();
  }
  
  const TypeDescription *ftd = hField->get_type_description(Field::FIELD_NAME_ONLY_E);
  const TypeDescription *mtd = hField->get_type_description(Field::MESH_TD_E);
  const TypeDescription *btd = hField->get_type_description(Field::BASIS_TD_E);
  const TypeDescription *dtd = hField->get_type_description(Field::FDATA_TD_E);

  CompileInfoHandle ci =
    ApplyFEMVoltageSourceAlgo::get_compile_info(ftd, mtd, btd, dtd);
  Handle<ApplyFEMVoltageSourceAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) 
    return;
  
  algo->execute(hField, rhsIn, matIn, bcFlag_.get(), mat, rhs);

  //! Sending result
  MatrixHandle mat_tmp(mat);
  send_output_handle("Forward Matrix", mat_tmp);

  MatrixHandle rhs_tmp(rhs);
  send_output_handle("RHS", rhs_tmp);
}


CompileInfoHandle
ApplyFEMVoltageSourceAlgo::get_compile_info(const TypeDescription *ftd,
			       const TypeDescription *mtd,
			       const TypeDescription *btd,
			       const TypeDescription *dtd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ApplyFEMVoltageSourceAlgoT");
  static const string base_class_name("ApplyFEMVoltageSourceAlgo");

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
