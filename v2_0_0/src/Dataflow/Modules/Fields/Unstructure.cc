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
 *  Unstructure: Store/retrieve values from an input matrix to/from 
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
#include <Dataflow/Modules/Fields/Unstructure.h>
#include <Core/Datatypes/StructHexVolField.h>
#include <Core/Datatypes/StructQuadSurfField.h>
#include <Core/Datatypes/StructCurveField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class Unstructure : public Module
{
public:
  Unstructure(GuiContext* ctx);
  virtual ~Unstructure();
  virtual void execute();

private:
  int last_generation_;
  FieldHandle ofieldhandle_;
};


DECLARE_MAKER(Unstructure)
Unstructure::Unstructure(GuiContext* ctx)
  : Module("Unstructure", ctx, Filter, "FieldsGeometry", "SCIRun"),
    last_generation_(0),
    ofieldhandle_(0)
{
}



Unstructure::~Unstructure()
{
}



void
Unstructure::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  if (ifieldhandle->generation != last_generation_)
  {
    last_generation_ = ifieldhandle->generation;
    ofieldhandle_ = ifieldhandle;
    string dstname = "";
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    const string &mtdn = mtd->get_name();
    if (mtdn == get_type_description((LatVolMesh *)0)->get_name() ||
	mtdn == get_type_description((StructHexVolMesh *)0)->get_name())
    {
      dstname = "HexVolField";
    }
    else if (mtdn == get_type_description((ImageMesh *)0)->get_name() ||
	     mtdn == get_type_description((StructQuadSurfMesh *)0)->get_name())
    {
      dstname = "QuadSurfField";
    }  
    else if (mtdn == get_type_description((ScanlineMesh *)0)->get_name() ||
	     mtdn == get_type_description((StructCurveMesh *)0)->get_name())
    {
      dstname = "CurveField";
    }

    if (dstname == "")
    {
      warning("Do not know how to unstructure a " + mtdn + ".");
    }
    else
    {
      const TypeDescription *ftd = ifieldhandle->get_type_description();
      CompileInfoHandle ci = UnstructureAlgo::get_compile_info(ftd, dstname);
      Handle<UnstructureAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;

      ofieldhandle_ = algo->execute(this, ifieldhandle);

      if (ofieldhandle_.get_rep())
      {
	*((PropertyManager *)(ofieldhandle_.get_rep())) =
	  *((PropertyManager *)(ifieldhandle.get_rep()));
      }
    }
  }

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port)
  {
    error("Unable to initialize " + name + "'s oport.");
    return;
  }
  ofield_port->send(ofieldhandle_);
}



CompileInfoHandle
UnstructureAlgo::get_compile_info(const TypeDescription *fsrc,
				  const string &partial_fdst)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("UnstructureAlgoT");
  static const string base_class_name("UnstructureAlgo");

  const string::size_type loc = fsrc->get_name().find_first_of('<');
  const string fdstname = partial_fdst + fsrc->get_name().substr(loc);

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       to_filename(fdstname) + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + "," + fdstname);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
