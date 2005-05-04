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

//    File   : ForwardMagneticField.cc
//    Author : Robert Van Uitert
//    Date   : Mon Aug  4 14:46:51 2003

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/BioPSE/Dataflow/Modules/Forward/ForwardMagneticField.h>

namespace BioPSE {

using namespace SCIRun;

DECLARE_MAKER(ForwardMagneticField)

ForwardMagneticField::ForwardMagneticField(GuiContext* ctx)
  : Module("ForwardMagneticField", ctx, Source, "Forward", "BioPSE")
{
}

ForwardMagneticField::~ForwardMagneticField(){
}

void ForwardMagneticField::execute(){
  

// J = (sigma)*E + J(source)
  electricFieldP_ = (FieldIPort *)get_iport("Electric Field");
  FieldHandle efld;
  if (!electricFieldP_->get(efld)) return;

  cond_tens_ = (FieldIPort *)get_iport("Conductivity Tensors");
  FieldHandle ctfld;
  if (!cond_tens_->get(ctfld)) return;

  if (efld->query_vector_interface().get_rep() == 0) {
    error("Must have Vector field as Electric Field input");
    return;
  }

  sourceLocationP_ = (FieldIPort *)get_iport("Dipole Sources");
  FieldHandle dipoles;
  if (!sourceLocationP_->get(dipoles)) return;

  if (dipoles->query_vector_interface().get_rep() == 0) {
    error("Must have Vector field as Dipole Sources input");
    return;
  }
  
  
  detectorPtsP_ = (FieldIPort *)get_iport("Detector Locations");
  FieldHandle detectors;
  if (!detectorPtsP_->get(detectors)) return;
  
  if (detectors->query_vector_interface().get_rep() == 0) {
    error("Must have Vector field as Detector Locations input");
    return;
  }

  FieldHandle magnetic_field;
  FieldHandle magnitudes;
  // create algo.
  const string new_field_type =
    detectors->get_type_description(0)->get_name() + "<double> ";  
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
  magneticFieldAtPointsP_ = (FieldOPort *)get_oport("Magnetic Field");
  magneticFieldAtPointsP_->send(magnetic_field);
  
  magnitudeFieldP_ = (FieldOPort *)get_oport("Magnitudes");
  magnitudeFieldP_->send(magnitudes);
  report_progress(ProgressReporter::Done);
}

void 
ForwardMagneticField::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

CalcFMFieldBase::~CalcFMFieldBase()
{
}

CompileInfoHandle
CalcFMFieldBase::get_compile_info(const TypeDescription *efldtd, 
				  const TypeDescription *ctfldtd,
				  const TypeDescription *detfldtd,
				  const string &mags) {
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
CalcFMFieldBase::get_h_file_path() {
  static const string path(TypeDescription::cc_to_h(__FILE__));
  return path;
}


} // End namespace BioPSE


