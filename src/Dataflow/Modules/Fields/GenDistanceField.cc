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
 *  GenDistanceField.cc:  Unfinished modules
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/GenDistanceField.h>
#include <iostream>


namespace SCIRun {

class GenDistanceField : public Module
{
private:

public:
  GenDistanceField(const string& id);
  virtual ~GenDistanceField();

  virtual void execute();
};


extern "C" Module* make_GenDistanceField(const string& id) {
  return new GenDistanceField(id);
}


GenDistanceField::GenDistanceField(const string& id)
  : Module("GenDistanceField", id, Filter, "Fields", "SCIRun")
{
}



GenDistanceField::~GenDistanceField()
{
}


void
GenDistanceField::execute()
{
  // Read in the LatticeVol<double>, clone it.
  FieldIPort *dst_port = (FieldIPort *)get_iport("Destination");
  if (!dst_port) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  FieldHandle dfieldhandle;
  if (!(dst_port->get(dfieldhandle) && dfieldhandle.get_rep()))
  {
    return;
  }
  if (!dfieldhandle->is_scalar())
  {
    error("Destination field must be of scalar type, preferably doubles.");
    return;
  }

  bool did_once_p = false;
  port_range_type range = get_iports("Skeleton");
  port_map_type::iterator pi = range.first;
  while (pi != range.second)
  {
    FieldIPort *port = (FieldIPort *)get_iport(pi->second);
    if (!port) {
      postMessage("Unable to initialize "+name+"'s iport\n");
      return;
    }
    ++pi;

    // Do something with port.
    FieldHandle skel_handle;
    if (port->get(skel_handle) && skel_handle.get_rep())
    {
      if (!skel_handle->is_scalar())
      {
	warning("Skeleton models must be scalar fields, skipping.");
	continue;
      }
      CompileInfo *ci =	GenDistanceFieldAlgo::
	get_compile_info(dfieldhandle->get_type_description(),
			 dfieldhandle->data_at_type_description(),
			 skel_handle->get_type_description());

      DynamicAlgoHandle algo_handle;
      if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
      {
	error("Could not compile algorithm.");
	continue;
      }
      GenDistanceFieldAlgo *algo =
	dynamic_cast<GenDistanceFieldAlgo *>(algo_handle.get_rep());
      if (algo == 0)
      {
	error("Could not get algorithm.");
	continue;
      }

      if (!did_once_p) { dfieldhandle.detach(); }
      if (skel_handle->data_at() == Field::NODE)
      {
	algo->execute_node(dfieldhandle, skel_handle, did_once_p);
      }
      else if (skel_handle->data_at() == Field::EDGE)
      {
	algo->execute_edge(dfieldhandle, skel_handle, did_once_p);
      }
      else
      {
	// TODO:  Could make execute_face and execute_cell functions.
	error("Skeleton data location must be at nodes or edges!");
	return;
      }
      did_once_p = true;
    }
  }
  
  if (did_once_p)
  {
    // Forward the lattice.
    FieldOPort *ofp = (FieldOPort *)get_oport("Distances");
    if (!ofp) {
      error("Unable to initialize " + name + "'s output port.");
      return;
    }
    ofp->send(dfieldhandle);
  }    
}


double
GenDistanceFieldAlgo::distance_to_line2(const Point &p,
					const Point &a, const Point &b) const
{
  Vector m = b - a;
  Vector n = p - a;
  const double t0 = Dot(m, n) / Dot(m, m);
  if (t0 <= 0) return (n).length2();
  else if (t0 >= 1.0) return (p - b).length2();
  else return (n - m * t0).length2();
}


CompileInfo *
GenDistanceFieldAlgo::get_compile_info(const TypeDescription *fdst,
				       const TypeDescription *ldst,
				       const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GenDistanceFieldAlgoT");
  static const string base_class_name("GenDistanceFieldAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fdst->get_filename() + "." +
		       ldst->get_filename() + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fdst->get_name() + ", " +
                       ldst->get_name() + ", " +
                       fsrc->get_name());
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fdst->fill_compile_info(rval);
  fsrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

