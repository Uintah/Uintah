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
 *  AttractNormals: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2002
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/AttractNormals.h>
#include <Core/Datatypes/PointCloud.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {


class PointAttractor : public Attractor
{
  Point point_;
public:
  PointAttractor(const Point &p) : point_(p) {}
  virtual void execute(Vector &v, const Point &p);
};

class LineAttractor : public Attractor
{
  Point point_;
  Vector direction_;

public:
  LineAttractor(const Point &p, const Vector &d) : point_(p), direction_(d)
  { direction_.normalize(); }
  virtual void execute(Vector &v, const Point &p);
};



class AttractNormals : public Module
{
public:
  AttractNormals(const string& id);
  virtual ~AttractNormals();
  virtual void execute();
};


extern "C" Module* make_AttractNormals(const string& id)
{
  return new AttractNormals(id);
}

AttractNormals::AttractNormals(const string& id)
  : Module("AttractNormals", id, Filter, "Fields", "SCIRun")
{
}



AttractNormals::~AttractNormals()
{
}



void
AttractNormals::execute()
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

  FieldIPort *ipp = (FieldIPort *)get_iport("Input Point");
  FieldHandle ipointhandle;
  if (!ipp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ipp->get(ipointhandle) && ipointhandle.get_rep()))
  {
    return;
  }
  
  PointCloudMesh *ipcm =
    dynamic_cast<PointCloudMesh *>(ipointhandle->mesh().get_rep());
  if (!ipcm)
  {
    error("Input point not in a PointCloud format.");
    return;
  }
  
  PointCloudMesh::Node::iterator itr, eitr;
  ipcm->begin(itr);
  ipcm->end(eitr);
  if (itr == eitr)
  {
    error("Empty PointCloud in Input Point Port.");
    return;
  }
  Point attract_point;
  ipcm->get_center(attract_point, *itr);

  AttractorHandle attractor = 0;
  PointCloud<Vector> *vpc =
    dynamic_cast<PointCloud<Vector> *>(ipointhandle.get_rep());
  if (vpc)
  {
    Vector dir;
    vpc->value(dir, *itr);
    attractor = scinew LineAttractor(attract_point, dir);
  }
  else
  {
    attractor = scinew PointAttractor(attract_point);
  }

  bool scale_p = false;
  if (ifieldhandle->query_scalar_interface())
  {
    scale_p = true;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  CompileInfo *ci =
    AttractNormalsAlgo::get_compile_info(ftd, ltd, mtd, scale_p);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  AttractNormalsAlgo *algo =
    dynamic_cast<AttractNormalsAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  FieldHandle ofieldhandle(algo->execute(ifieldhandle, attractor));

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  ofield_port->send(ofieldhandle);
}


void
Attractor::execute(Vector &v, const Point &p)
{
  v = p.asVector();
}

void
Attractor::io(Piostream &s)
{
}


void
PointAttractor::execute(Vector &v, const Point &p)
{
  v = point_ - p;
  v.safe_normalize();
}


void
LineAttractor::execute(Vector &v, const Point &p)
{
  Vector diff = p - point_;
  Vector proj(direction_ * Dot(direction_, diff));
  
  v = proj - diff;

  v.safe_normalize();
}



CompileInfo *
AttractNormalsAlgo::get_compile_info(const TypeDescription *fsrc_td,
				     const TypeDescription *floc_td,
				     const TypeDescription *msrc_td,
				     bool scale_p)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("AttractNormalsAlgoT");
  static const string scale_template_class_name("AttractNormalsScaleAlgoT");
  static const string base_class_name("AttractNormalsAlgo");
  const string::size_type fsrc_loc = fsrc_td->get_name().find_first_of('<');
  const string fdst = fsrc_td->get_name().substr(0, fsrc_loc) + "<Vector> ";

  CompileInfo *rval;

  if (scale_p)
  {
    rval = 
      scinew CompileInfo(scale_template_class_name + "." +
			 fsrc_td->get_filename() + "." +
			 floc_td->get_filename() + ".",
			 base_class_name, 
			 scale_template_class_name, 
			 fsrc_td->get_name() + ", " +
			 floc_td->get_name() + ", " + fdst);
  }
  else
  {
    rval = 
      scinew CompileInfo(template_class_name + "." +
			 msrc_td->get_filename() + "." +
			 floc_td->get_filename() + ".",
			 base_class_name, 
			 template_class_name, 
			 msrc_td->get_name() + ", " +
			 floc_td->get_name() + ", " + fdst);
  }

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
