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
 *  MoveElemToNode.cc:  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/MoveElemToNode.h>
#include <iostream>

namespace SCIRun {


class MoveElemToNode : public Module
{
public:
  MoveElemToNode(GuiContext* ctx);
  virtual ~MoveElemToNode();

  virtual void execute();

protected:
  int ifield_generation_;
};


DECLARE_MAKER(MoveElemToNode)

MoveElemToNode::MoveElemToNode(GuiContext* ctx)
  : Module("MoveElemToNode", ctx, Filter, "Fields", "SCIRun"),
    ifield_generation_(0)
{
}


MoveElemToNode::~MoveElemToNode()
{
}


void
MoveElemToNode::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Elem Field");
  FieldHandle ifield;
  if (!ifp) {
    error("Unable to initialize iport 'Elem Field'.");
    return;
  }
  if (!(ifp->get(ifield) && ifield.get_rep()))
  {
    return;
  }

  string ext = "Lat";

  if (ifield_generation_ != ifield->generation)
  {
    const TypeDescription *ftd = ifield->get_type_description();
    CompileInfoHandle ci = MoveElemToNodeAlgo::get_compile_info(ftd, ext);
    Handle<MoveElemToNodeAlgo> algo;
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Unable to compile MoveElemToNode algorithm.");
      return;
    }

    FieldHandle ofield = algo->execute(this, ifield);
  
    FieldOPort *ofp = (FieldOPort *)get_oport("Node Field");
    if (!ofp) {
      error("Unable to initialize oport 'Node Field'.");
      return;
    }

    ofp->send(ofield);
  }
}


CompileInfoHandle
MoveElemToNodeAlgo::get_compile_info(const TypeDescription *fsrc, string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("MoveElemToNodeAlgo" + ext);
  static const string base_class_name("MoveElemToNodeAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);

  return rval;
}


} // End namespace SCIRun

