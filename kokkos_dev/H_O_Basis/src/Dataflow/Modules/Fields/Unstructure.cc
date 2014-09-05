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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/Unstructure.h>
#include <Core/Datatypes/StructHexVolMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Datatypes/StructCurveMesh.h>
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

protected:
  GuiInt gPointCloud_;
  unsigned int pointCloud_;

  int last_generation_;
  FieldHandle ofieldhandle_;
};


DECLARE_MAKER(Unstructure)
Unstructure::Unstructure(GuiContext* context)
  : Module("Unstructure", context, Filter, "FieldsGeometry", "SCIRun"),
    gPointCloud_(context->subVar("point-cloud")),
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
  bool update = false;

  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;

  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep())) {
    error( "No handle or representation" );
    return;
  }

  if( pointCloud_ != (unsigned int) gPointCloud_.get() ) {
    update = true;
    pointCloud_ = gPointCloud_.get();
  }

  if (ifieldhandle->generation != last_generation_) {
    update = true;
    last_generation_ = ifieldhandle->generation;
  }

  if( update ) {
    string dstname("");
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    const string &mtdn = mtd->get_name();
<<<<<<< .working
    if (mtdn == get_type_description((LVMesh *)0)->get_name() ||
	mtdn == get_type_description((SHVolMesh *)0)->get_name())
    {
=======

    if( pointCloud_ )
      dstname = "PointCloudField";
    else if (mtdn == "LatVolMesh"   || mtdn == "StructHexVolMesh")
>>>>>>> .merge-right.r32054
      dstname = "HexVolField";
<<<<<<< .working
    }
    else if (mtdn == get_type_description((IMesh *)0)->get_name() ||
	     mtdn == get_type_description((SQSMesh *)0)->get_name())
    {
=======
    else if (mtdn == "ImageMesh"    || mtdn == "StructQuadSurfMesh")
>>>>>>> .merge-right.r32054
      dstname = "QuadSurfField";
<<<<<<< .working
    }  
    else if (mtdn == get_type_description((SLMesh *)0)->get_name() ||
	     mtdn == get_type_description((SCMesh *)0)->get_name())
    {
=======
    else if (mtdn == "ScanlineMesh" || mtdn == "StructCurveMesh" )
>>>>>>> .merge-right.r32054
      dstname = "CurveField";
  
    if (dstname == "") {
      warning("Do not know how to unstructure a " + mtdn + ".");

      ofieldhandle_ = ifieldhandle;
    }
    else {
      const TypeDescription *ftd = ifieldhandle->get_type_description();
      CompileInfoHandle ci = UnstructureAlgo::get_compile_info(ftd, dstname);
      Handle<UnstructureAlgo> algo;

      if (!module_dynamic_compile(ci, algo)) return;

      ofieldhandle_ = algo->execute(this, ifieldhandle);

      if (ofieldhandle_.get_rep())
	ofieldhandle_->copy_properties(ifieldhandle.get_rep());
    }
  }

  // Get a handle to the output field port.
  if ( ofieldhandle_.get_rep() ) {
    FieldOPort* ofp = (FieldOPort *) get_oport("Output Field");

    // Send the data downstream
    ofp->send(ofieldhandle_);
  }
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
