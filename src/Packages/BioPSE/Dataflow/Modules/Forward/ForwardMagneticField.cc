//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
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


