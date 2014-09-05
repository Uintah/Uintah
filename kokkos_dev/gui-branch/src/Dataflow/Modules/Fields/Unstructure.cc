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
#include <Core/Parts/GuiVar.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class Unstructure : public Module
{
public:
  Unstructure(const string& id);
  virtual ~Unstructure();
  virtual void execute();

private:
  int last_generation_;
  FieldHandle ofieldhandle_;
};


extern "C" Module* make_Unstructure(const string& id)
{
  return new Unstructure(id);
}

Unstructure::Unstructure(const string& id)
  : Module("Unstructure", id, Filter, "Fields", "SCIRun"),
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
    postMessage("Unable to initialize "+name+"'s iport\n");
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
    if (mtd->get_name() == get_type_description((LatVolMesh *)0)->get_name())
    {
      dstname = "HexVol";
    }
    if (mtd->get_name() == get_type_description((ImageMesh *)0)->get_name())
    {
      dstname = "QuadSurf";
    }  
    if (mtd->get_name() == get_type_description((ScanlineMesh *)0)->get_name())
    {
      dstname = "ContourField";
    }

    if (dstname == "")
    {
      warning("Do not know how to unstructure a " + mtd->get_name() + ".");
    }
    else
    {
      const TypeDescription *ftd = ifieldhandle->get_type_description();
      CompileInfo *ci = UnstructureAlgo::get_compile_info(ftd, dstname);
      DynamicAlgoHandle algo_handle;
      if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
      {
	error("Could not compile algorithm.");
	return;
      }
      UnstructureAlgo *algo =
	dynamic_cast<UnstructureAlgo *>(algo_handle.get_rep());
      if (algo == 0)
      {
	error("Could not get algorithm.");
	return;
      }
      ofieldhandle_ = algo->execute(ifieldhandle);
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



CompileInfo *
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
