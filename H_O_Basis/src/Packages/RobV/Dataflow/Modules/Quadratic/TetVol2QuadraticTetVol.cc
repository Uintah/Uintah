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
//    File   : TetVol2QuadraticTetVol.cc
//    Author : Martin Cole
//    Date   : Wed Feb 27 09:01:36 2002

#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/QuadraticTetVolField.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/RobV/Dataflow/Modules/Quadratic/ConvertTet.h>

#include <Packages/RobV/share/share.h>

namespace RobV {

using namespace SCIRun;

class RobVSHARE TetVol2QuadraticTetVol : public Module {
public:
  TetVol2QuadraticTetVol(const string& id);

  virtual ~TetVol2QuadraticTetVol();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);

private:
  FieldIPort*              ifld_;
  FieldOPort*              ofld_;
};

extern "C" RobVSHARE Module* make_TetVol2QuadraticTetVol(const string& id) {
  return scinew TetVol2QuadraticTetVol(id);
}

TetVol2QuadraticTetVol::TetVol2QuadraticTetVol(const string& id)
  : Module("TetVol2QuadraticTetVol", id, Source, "Quadratic", "RobV")
{
}

TetVol2QuadraticTetVol::~TetVol2QuadraticTetVol(){
}

void TetVol2QuadraticTetVol::execute()
{
  ifld_ = (FieldIPort *)get_iport("InputTetField");
  FieldHandle fld_handle;
  
  ifld_->get(fld_handle);

  if(!fld_handle.get_rep()){
    warning("No Data in port 1 field.");
    return;
  } else if (fld_handle->get_type_name(0) != "TetVol") {
    error("input must be a TetVol type, not a "+fld_handle->get_type_name(0));
    return;
  }
  const TypeDescription *td = fld_handle->get_type_description();
  CompileInfo *ci = ConvertTetBase::get_compile_info(td);
  DynamicAlgoHandle  converter;
  if (! DynamicLoader::scirun_loader().get(*ci, converter)) {
    error("Could not compile algorithm for TetVol2QuadraticTetVol -");
    error(td->get_name().c_str());
    return;
  }

  if (converter.get_rep() == 0) {
    error("TetVol2QuadraticTetVol could not get algorithm!!");
    return;
  }
  ConvertTetBase *conv = (ConvertTetBase*)converter.get_rep();


  FieldHandle ofld_handle = conv->convert_quadratic(fld_handle);
  ofld_ = (FieldOPort *)get_oport("OutputQuadraticTet");
  ofld_->send(ofld_handle);
}

void TetVol2QuadraticTetVol::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace RobV


