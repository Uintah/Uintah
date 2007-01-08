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

//    File   : SimulateForwardMagneticField.cc
//    Author : Robert Van Uitert
//    Date   : Mon Aug  4 14:46:51 2003

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/BioPSE/Dataflow/Modules/Forward/SimulateForwardMagneticField.h>

namespace BioPSE {

using namespace SCIRun;

class SimulateForwardMagneticField : public Module {
public:
  SimulateForwardMagneticField(GuiContext *context);

  virtual ~SimulateForwardMagneticField();

  virtual void execute();
};


DECLARE_MAKER(SimulateForwardMagneticField)

SimulateForwardMagneticField::SimulateForwardMagneticField(GuiContext* ctx)
  : Module("SimulateForwardMagneticField", ctx, Source, "Forward", "BioPSE")
{
}

SimulateForwardMagneticField::~SimulateForwardMagneticField()
{
}


void
SimulateForwardMagneticField::execute()
{
// J = (sigma)*E + J(source)
  FieldHandle efld;
  if (!get_input_handle("Electric Field", efld)) return;

  if (efld->query_vector_interface().get_rep() == 0) {
    error("Must have Vector field as Electric Field input");
    return;
  }

  FieldHandle ctfld;
  if (!get_input_handle("Conductivity Tensors", ctfld)) return;

  FieldHandle dipoles;
  if (!get_input_handle("Dipole Sources", dipoles)) return;

  if (dipoles->query_vector_interface().get_rep() == 0) {
    error("Must have Vector field as Dipole Sources input");
    return;
  }
  
  FieldHandle detectors;
  if (!get_input_handle("Detector Locations", detectors)) return;
  
  if (detectors->query_vector_interface().get_rep() == 0) {
    error("Must have Vector field as Detector Locations input");
    return;
  }

  FieldHandle magnetic_field;
  FieldHandle magnitudes;
  // create algo.
  const string new_field_type =
    detectors->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() + "<double> ";  
  const TypeDescription *efld_td = efld->get_type_description();
  const TypeDescription *ctfld_td = ctfld->get_type_description();
  const TypeDescription *detfld_td = detectors->get_type_description();
  CompileInfoHandle ci = CalcFMFieldBase::get_compile_info(efld_td, ctfld_td, 
							   detfld_td, 
							   new_field_type);
  Handle<CalcFMFieldBase> algo;
  if (!DynamicCompilation::compile(ci, algo, this))
  {
    error("Unable to compile creation algorithm.");
    return;
  }
  update_progress(0);
  int np = 5;
  //cerr << "Number of Processors Used: " << np <<endl;
  if (! algo->calc_forward_magnetic_field(efld, ctfld, dipoles, detectors, 
					  magnetic_field, magnitudes, np, 
					  this)) {

    return;
  }
  
  if (magnetic_field.get_rep() == 0) {
    error("No field to output (Magnetic Field)");
    cerr << "null field magnetic" << endl;
    return;
  }
    
  if (magnitudes.get_rep() == 0) {
    error("No field to output (Magnitudes)");
     cerr << "null field magnitudes" << endl;
    return;
  }

  send_output_handle("Magnetic Field", magnetic_field);
  
  send_output_handle("Magnitudes", magnitudes);
}


CalcFMFieldBase::~CalcFMFieldBase()
{
}

CompileInfoHandle
CalcFMFieldBase::get_compile_info(const TypeDescription *efldtd, 
				  const TypeDescription *ctfldtd,
				  const TypeDescription *detfldtd,
				  const string &mags)
{
  static const string ns("BioPSE");
  static const string template_class("CalcFMField");
  CompileInfo *rval = scinew CompileInfo(template_class + "." +
					 efldtd->get_filename() + "." +
					 ctfldtd->get_filename() + "." +
					 detfldtd->get_filename() + "." +
					 to_filename(mags) + ".",
					 base_class_name(), 
					 template_class, 
					 efldtd->get_name() + "," + 
					 ctfldtd->get_name() + "," + 
					 detfldtd->get_name() + "," + 
					 mags + " ");
  rval->add_include(get_h_file_path());
  rval->add_namespace(ns);
  efldtd->fill_compile_info(rval);
  ctfldtd->fill_compile_info(rval);
  detfldtd->fill_compile_info(rval);
  return rval;
}

const string& 
CalcFMFieldBase::get_h_file_path()
{
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}


} // End namespace BioPSE


