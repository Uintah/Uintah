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
 *  ApplyInterpolant.cc:  Apply an interpolant field to project the data
 *                 from one field onto the mesh of another field.
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parts/GuiVar.h>
#include <Dataflow/Modules/Fields/ApplyInterpolant.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class ApplyInterpolant : public Module {
private:
  FieldIPort *src_port;
  FieldIPort *itp_port;
  FieldOPort *ofp;
public:
  ApplyInterpolant(const string& id);
  virtual ~ApplyInterpolant();

  virtual void execute();
};


extern "C" Module* make_ApplyInterpolant(const string& id)
{
  return new ApplyInterpolant(id);
}


ApplyInterpolant::ApplyInterpolant(const string& id)
  : Module("ApplyInterpolant", id, Filter, "Fields", "SCIRun")
{
}


ApplyInterpolant::~ApplyInterpolant()
{
}



void
ApplyInterpolant::execute()
{
  src_port = (FieldIPort *)get_iport("Source");
  FieldHandle fsrc_h;

  if(!src_port) {
    postMessage("Unable to initialize "+name+"'s iport");
    return;
  }
  if (!(src_port->get(fsrc_h) && fsrc_h.get_rep()))
  {
    return;
  }

  itp_port = (FieldIPort *)get_iport("Interpolant");
  FieldHandle fitp_h;

  if (!itp_port) {
    postMessage("Unable to initialize "+name+"'s iport"); 
    return;
  }
  if (!(itp_port->get(fitp_h) && fitp_h.get_rep()))
  {
    return;
  }

  CompileInfo *ci =
    ApplyInterpAlgo::get_compile_info(fsrc_h->get_type_description(),
				      fitp_h->get_type_description(),
				      fitp_h->data_at_type_description());
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    error("Could not compile algorithm.");
    return;
  }
  ApplyInterpAlgo *algo =
    dynamic_cast<ApplyInterpAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    error("Could not get algorithm.");
    return;
  }

  ofp = (FieldOPort *)get_oport("Output");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
    
  ofp->send(algo->execute(fsrc_h, fitp_h));
}



CompileInfo *
ApplyInterpAlgo::get_compile_info(const TypeDescription *fsrc,
				  const TypeDescription *fitp,
				  const TypeDescription *litp)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ApplyInterpAlgoT");
  static const string base_class_name("ApplyInterpAlgo");

  const string::size_type fitp_loc = fitp->get_name().find_first_of('<');
  const string::size_type fsrc_loc = fsrc->get_name().find_first_of('<');
  const string fout = fitp->get_name().substr(0, fitp_loc) +
    fsrc->get_name().substr(fsrc_loc);

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       fitp->get_filename() + "." +
		       litp->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + ", " +
                       fitp->get_name() + ", " +
                       litp->get_name() + ", " +
                       fout);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  fitp->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


