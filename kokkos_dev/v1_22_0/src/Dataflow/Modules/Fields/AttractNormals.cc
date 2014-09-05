/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
#include <Core/Datatypes/PointCloudField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/DynamicCompilation.h>

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
  AttractNormals(GuiContext* ctx);
  virtual ~AttractNormals();
  virtual void execute();
};


DECLARE_MAKER(AttractNormals)
AttractNormals::AttractNormals(GuiContext* ctx)
  : Module("AttractNormals", ctx, Filter, "FieldsData", "SCIRun")
{
}



AttractNormals::~AttractNormals()
{
}



void
AttractNormals::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)getIPort("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  FieldIPort *ipp = (FieldIPort *)get_iport("Input Point");
  FieldHandle ipointhandle;
  if (!ipp) {
    error("Unable to initialize iport 'Input Point'.");
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
    error("Input point not in a PointCloudField format.");
    return;
  }
  
  PointCloudMesh::Node::iterator itr, eitr;
  ipcm->begin(itr);
  ipcm->end(eitr);
  if (itr == eitr)
  {
    error("Empty PointCloudField in Input Point Port.");
    return;
  }
  Point attract_point;
  ipcm->get_center(attract_point, *itr);

  AttractorHandle attractor = 0;
  PointCloudField<Vector> *vpc =
    dynamic_cast<PointCloudField<Vector> *>(ipointhandle.get_rep());
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
  ScalarFieldInterfaceHandle sfi = ifieldhandle->query_scalar_interface(this);
  if (sfi.get_rep())
  {
    scale_p = true;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  CompileInfoHandle ci =
    AttractNormalsAlgo::get_compile_info(ftd, ltd, mtd, scale_p);
  Handle<AttractNormalsAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  FieldHandle ofieldhandle(algo->execute(ifieldhandle, attractor));

  FieldOPort *ofield_port = (FieldOPort *)getOPort("Output Field");
  if (!ofield_port) {
    error("Unable to initialize oport 'Output Field'.");
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
Attractor::io(Piostream &)
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



CompileInfoHandle
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
