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
 *  MapDistanceField.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   David Weinstein
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Butson/Dataflow/Modules/Modeling/MapDistanceField.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

using std::vector;
using std::pair;


class MapDistanceField : public Module
{
  FieldIPort *src_port;
  FieldIPort *dst_port;
  FieldOPort *ofp; 
  GuiString   interp_op_gui_;

public:
  MapDistanceField(GuiContext *context);
  virtual ~MapDistanceField();
  virtual void execute();

  //template <class Mesh, class Index>
  //void find_closest(Mesh *mesh, typename Index::index_type &idx, Point &p);
};


DECLARE_MAKER(MapDistanceField)


MapDistanceField::MapDistanceField(GuiContext *context) : 
  Module("MapDistanceField", context, Filter, "Modeling", "Butson"),
  interp_op_gui_(context->subVar("interp_op_gui"))
{
}

MapDistanceField::~MapDistanceField()
{
}



void
MapDistanceField::execute()
{
  dst_port = (FieldIPort *)get_iport("Surface");
  FieldHandle fdst_h;

  if (!dst_port) {
    error("Unable to initialize iport 'Surface'.");
    return;
  }
  if (!(dst_port->get(fdst_h) && fdst_h.get_rep()))
  {
    return;
  }

  src_port = (FieldIPort *)get_iport("Contour");
  FieldHandle fsrc_h;
  if(!src_port) {
    error("Unable to initialize iport 'Contour'.");
    return;
  }
  if (!(src_port->get(fsrc_h) && fsrc_h.get_rep()))
  {
    return;
  }

  CompileInfoHandle ci =
    MapDistanceFieldAlgo::get_compile_info(fsrc_h->get_type_description(),
					   fsrc_h->data_at_type_description(),
					   fdst_h->mesh()->get_type_description(),
					   fdst_h->data_at_type_description(),
					   fdst_h->get_type_description());
  Handle<MapDistanceFieldAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  pair<FieldHandle, FieldHandle> result;
  result = algo->execute(fsrc_h, fdst_h->mesh(), fdst_h->data_at());

  FieldOPort *ofp1 = (FieldOPort *)get_oport("From Contour Interpolant");
  if(!ofp1) {
    error("Unable to initialize oport 'From Contour Interpolant'.");
    return;
  }
  ofp1->send(result.second);

  FieldOPort *ofp2 = (FieldOPort *)get_oport("From Surface Interpolant");
  if(!ofp2) {
    error("Unable to initialize oport 'From Surface Interpolant'.");
    return;
  }
  ofp2->send(result.first);
}


double
MapDistanceFieldAlgo::distance_to_line2(const Point &p,
					const Point &a, const Point &b) const
{
  Vector m = b - a;
  Vector n = p - a;
  const double t0 = Dot(m, n) / Dot(m, m);
  if (t0 <= 0) return (n).length2();
  else if (t0 >= 1.0) return (p - b).length2();
  else return (n - m * t0).length2();
}


CompileInfoHandle
MapDistanceFieldAlgo::get_compile_info(const TypeDescription *fsrc,
				       const TypeDescription *lsrc,
				       const TypeDescription *mdst,
				       const TypeDescription *ldst,
				       const TypeDescription *fdst)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("MapDistanceFieldAlgoT");
  static const string base_class_name("MapDistanceFieldAlgo");

  const string::size_type loc1 = fsrc->get_name().find_first_of('<');
  const string foutsrc = fsrc->get_name().substr(0, loc1) +
    "<vector<pair<" + ldst->get_name() + "::index_type, double> > > ";

  const string::size_type loc2 = fdst->get_name().find_first_of('<');
  const string foutdst = fdst->get_name().substr(0, loc2) +
    "<vector<pair<" + lsrc->get_name() + "::index_type, double> > > ";

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       lsrc->get_filename() + "." +
		       fdst->get_filename() + "." +
		       ldst->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + ", " +
                       lsrc->get_name() + ", " +
                       mdst->get_name() + ", " +
                       ldst->get_name() + ", " +
                       foutsrc + ", " +
		       foutdst);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  fdst->fill_compile_info(rval);
  return rval;
}




} // End namespace SCIRun
