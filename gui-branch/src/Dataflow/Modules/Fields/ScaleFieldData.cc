/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  ScaleFieldData: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ScaleFieldData.h>
#include <Core/Parts/GuiVar.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ScaleFieldData : public Module
{
public:
  ScaleFieldData(const string& id);
  virtual ~ScaleFieldData();
  virtual void execute();
};


extern "C" Module* make_ScaleFieldData(const string& id)
{
  return new ScaleFieldData(id);
}

ScaleFieldData::ScaleFieldData(const string& id)
  : Module("ScaleFieldData", id, Filter, "Fields", "SCIRun")
{
}



ScaleFieldData::~ScaleFieldData()
{
}



void
ScaleFieldData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrix;
  if (!imatrix_port) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!imatrix_port->get(imatrix))
  {
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  CompileInfo *ci = ScaleFieldDataAlgo::get_compile_info(ftd, ltd);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  ScaleFieldDataAlgo *algo =
    dynamic_cast<ScaleFieldDataAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  FieldHandle ofieldhandle(algo->execute(ifieldhandle, imatrix));

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  ofield_port->send(ofieldhandle);
}



CompileInfo *
ScaleFieldDataAlgo::get_compile_info(const TypeDescription *field_td,
				     const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScaleFieldDataAlgoT");
  static const string base_class_name("ScaleFieldDataAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name() + ", " + loc_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
