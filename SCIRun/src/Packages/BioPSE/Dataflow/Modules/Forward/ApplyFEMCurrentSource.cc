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
 *  ApplyFEMCurrentSource.cc:  Builds the RHS of the FE matrix for voltage sources
 *
 *  Written by:
 *   David Weinstein, University of Utah, May 1999
 *   Alexei Samsonov, March 2001
 *   Frank B. Sachse, February 2006
 */

#include <Packages/BioPSE/Dataflow/Modules/Forward/ApplyFEMCurrentSource.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

namespace BioPSE {

class ApplyFEMCurrentSource : public Module
{
  GuiInt sourceNodeTCL_;
  GuiInt sinkNodeTCL_;
  GuiString modeTCL_;

public:
  ApplyFEMCurrentSource(GuiContext *context);
  virtual ~ApplyFEMCurrentSource();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(ApplyFEMCurrentSource)


ApplyFEMCurrentSource::ApplyFEMCurrentSource(GuiContext *context) : 
  Module("ApplyFEMCurrentSource", context, Filter, "Forward", "BioPSE"),
    sourceNodeTCL_(context->subVar("sourceNodeTCL")),
    sinkNodeTCL_(context->subVar("sinkNodeTCL")),
    modeTCL_(context->subVar("modeTCL"))      
{
  //  cerr << "ApplyFEMCurrentSource" << endl;
}


ApplyFEMCurrentSource::~ApplyFEMCurrentSource()
{
}


void
ApplyFEMCurrentSource::execute()
{ 
  bool dipole=false;

  if (modeTCL_.get() == "dipole")
    dipole=true;
  else if (modeTCL_.get() == "sources and sinks")
    dipole=false;
  else
    error("Unreachable code, bad mode.");

  // Get the input mesh.
  FieldHandle hField;
  if (!get_input_handle("Mesh", hField)) return;

  // Get the input dipoles.
  FieldHandle hSource;
  get_input_handle("Sources", hSource, false);
	
  // If the user passed in a vector the right size, copy it into ours.
  ColumnMatrix* rhs = 0;
  MatrixHandle  hRhsIn;
  if (get_input_handle("Input RHS", hRhsIn, false))
  {
    rhs = scinew ColumnMatrix(hRhsIn->nrows());
    string units;
    if (hRhsIn->get_property("units", units))
      rhs->set_property("units", units, false);
    for (int i=0; i < hRhsIn->nrows(); i++) 
      rhs->put(i, hRhsIn->get(i, 0));
  }

  MatrixHandle hMapping;
  if (!dipole) {
    get_input_handle("Mapping", hMapping, false);
  }  

  const TypeDescription *ftd = hField->get_type_description(Field::FIELD_NAME_ONLY_E);
  const TypeDescription *mtd = hField->get_type_description(Field::MESH_TD_E);
  const TypeDescription *btd = hField->get_type_description(Field::BASIS_TD_E);
  const TypeDescription *dtd = hField->get_type_description(Field::FDATA_TD_E);

  CompileInfoHandle ci =
    ApplyFEMCurrentSourceAlgo::get_compile_info(ftd, mtd, btd, dtd);
  Handle<ApplyFEMCurrentSourceAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) 
    return;

  SparseRowMatrix *w=NULL;
  
  algo->execute(this, hField, hSource, hMapping, dipole,
                Max(sourceNodeTCL_.get(),0), Max(sinkNodeTCL_.get(),0),
                &rhs, &w);

  //! Sending result
  MatrixHandle rhs_tmp(rhs);
  send_output_handle("Output RHS", rhs_tmp);

  if (w)
  {
    MatrixHandle wtmp(w);
    send_output_handle("Output Weights", wtmp);
  }
}


CompileInfoHandle
ApplyFEMCurrentSourceAlgo::get_compile_info(const TypeDescription *ftd,
			       const TypeDescription *mtd,
			       const TypeDescription *btd,
			       const TypeDescription *dtd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ApplyFEMCurrentSourceAlgoT");
  static const string base_class_name("ApplyFEMCurrentSourceAlgo");

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
