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
 *  QuadToTri.cc:  Convert a Quad field into a Tri field using 1-5 split
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Dataflow/Modules/Fields/QuadToTri.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <vector>
#include <algorithm>


namespace SCIRun {

class QuadToTri : public Module {
private:
  int last_generation_;
  FieldHandle ofieldhandle;

public:

  //! Constructor/Destructor
  QuadToTri(GuiContext *context);
  virtual ~QuadToTri();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(QuadToTri)


QuadToTri::QuadToTri(GuiContext *context) : 
  Module("QuadToTri", context, Filter, "Fields", "SCIRun"),
  last_generation_(0)
{
}

QuadToTri::~QuadToTri()
{
}

void
QuadToTri::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("QuadSurf");
  if (!ifieldport) {
    error("Unable to initialize iport 'QuadSurf'.");
    return;
  }
  FieldHandle ifieldhandle;
  if(!(ifieldport->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Can't get field.");
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("TriSurf");
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
  
  const TypeDescription *src_td = ifieldhandle->get_type_description();

  const string iname =
    ifieldhandle->mesh()->get_type_description()->get_name();
  if (iname == "ImageMesh" || iname == "StructQuadSurfMesh")
  {
    CompileInfoHandle ici = ImgToTriAlgo::get_compile_info(src_td);
    Handle<ImgToTriAlgo> ialgo;
    if (DynamicCompilation::compile(ici, ialgo, true, this))
    {
      if (!ialgo->execute(ifieldhandle, ofieldhandle, this))
      {
	warning("ImgToTri conversion failed to copy data.");
	return;
      }
    }
    else
    {
      error("QuadToTri only supports Quad field types.");
      return;
    }
  }
  else
  {
    CompileInfoHandle qci = QuadToTriAlgo::get_compile_info(src_td);
    Handle<QuadToTriAlgo> qalgo;
    if (DynamicCompilation::compile(qci, qalgo, true, this))
    {
      if (!qalgo->execute(ifieldhandle, ofieldhandle, this))
      {
	warning("QuadToTri conversion failed to copy data.");
	return;
      }
    }
    else
    {
      error("QuadToTri only supports Quad field types.");
      return;
    }
  }
  ofp->send(ofieldhandle);
}

CompileInfoHandle
QuadToTriAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("QuadToTriAlgoT");
  static const string base_class_name("QuadToTriAlgo");

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
ImgToTriAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ImgToTriAlgoT");
  static const string base_class_name("ImgToTriAlgo");

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
