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
 *  HexToTet.cc:  Convert a Hex field into a Tet field using 1-5 split
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Modules/Fields/HexToTet.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DynamicCompilation.h>

#include <iostream>
#include <vector>
#include <algorithm>


namespace SCIRun {

class HexToTet : public Module {
private:
  int last_generation_;
  FieldHandle ofieldhandle;

public:

  //! Constructor/Destructor
  HexToTet(GuiContext *context);
  virtual ~HexToTet();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(HexToTet)


HexToTet::HexToTet(GuiContext *context) : 
  Module("HexToTet", context, Filter, "Fields", "SCIRun"),
  last_generation_(0)
{
}

HexToTet::~HexToTet()
{
}

void
HexToTet::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("HexVol");
  if (!ifieldport) {
    error("Unable to initialize iport 'HexVol'.");
    return;
  }
  FieldHandle ifieldhandle;
  if(!(ifieldport->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Can't get field.");
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("TetVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }

  // Cache generation.
  if (ifieldhandle->generation == last_generation_)
  {
    ofp->send(ofieldhandle);
    return;
  }
  last_generation_ = ifieldhandle->generation;
  std::ostream &msg = msgStream();
  const TypeDescription *src_td = ifieldhandle->get_type_description();
  CompileInfoHandle hci = HexToTetAlgo::get_compile_info(src_td);
  Handle<HexToTetAlgo> halgo;
  if (DynamicCompilation::compile(hci, halgo, true, this))
  {
    if (!halgo->execute(ifieldhandle, ofieldhandle, msg))
    {
      warning("HexToTet conversion failed to copy data.");
      return;
    }
  }
  else
  {
    CompileInfoHandle lci = LatToTetAlgo::get_compile_info(src_td);
    Handle<LatToTetAlgo> lalgo;
    if (DynamicCompilation::compile(lci, lalgo, true, this))
    {
      if (!lalgo->execute(ifieldhandle, ofieldhandle, msg))
      {
	warning("LatToTet conversion failed to copy data.");
	return;
      }
    }
    else
    {
      error("HexToTet only supported for Hex types -- failed for "+
	    src_td->get_name());
      return;
    }
  }
  msgStream_flush();
  ofp->send(ofieldhandle);
}


CompileInfoHandle
HexToTetAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("HexToTetAlgoT");
  static const string base_class_name("HexToTetAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       src_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       src_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  src_td->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
LatToTetAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("LatToTetAlgoT");
  static const string base_class_name("LatToTetAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       src_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       src_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  src_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
