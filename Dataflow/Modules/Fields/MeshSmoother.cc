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
 *  MeshSmoother.cc:  Smooth a given mesh.
 *
 *  Written by:
 *   Jason Shepherd
 *   Department of Computer Science
 *   University of Utah
 *   January 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/MeshSmoother.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

class MeshSmoother : public Module
{
private:
  int       last_field_generation_;

public:
  MeshSmoother(GuiContext* ctx);
  virtual ~MeshSmoother();

  virtual void execute();
};


DECLARE_MAKER(MeshSmoother)


MeshSmoother::MeshSmoother(GuiContext* ctx)
        : Module("MeshSmoother", ctx, Filter, "FieldsData", "SCIRun"),
          last_field_generation_(0)
{
}

MeshSmoother::~MeshSmoother()
{
}

void MeshSmoother::execute()
{
    // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input");
  FieldHandle ifieldhandle;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  if (last_field_generation_ == ifieldhandle->generation &&
      oport_cached("Smoothed") )
  {
    // We're up to date, return.
    return;
  }
  last_field_generation_ = ifieldhandle->generation;

  cout << "Smoothing hexesa..." << endl;
  
  string ext = "";
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  if (mtd->get_name().find("TetVolMesh") != string::npos)
  {
    ext = "Tet";
  }
  else if (mtd->get_name().find("TriSurfMesh") != string::npos)
  {
//    ext = "Tri";
    error( "TriSurfMesh Fields are not currently supported by the MeshSmoother.");
    return;
  }
  else if (mtd->get_name().find("HexVolMesh") != string::npos)
  {
    ext = "Hex";
//    error("HexVolFields are not directly supported in this module.  Please first convert it into a TetVolField by inserting a SCIRun::FieldsGeometry::HexToTet module upstream.");
//    return;
  }
  else if (mtd->get_name().find("QuadSurfMesh") != string::npos)
  {
    error("QuadSurfFields are not currently supported in the MeshSmoother  module.");
    return;
  }
  else if (mtd->get_name().find("LatVolMesh") != string::npos)
  {
    error("LatVolFields are not directly supported in this module.  Please first convert it into a HexVolField by inserting an upstream SCIRun::FieldsGeometry::Unstructure module.");
    return;
  }
  else if (mtd->get_name().find("ImageMesh") != string::npos)
  {
    error("ImageFields are not currently supported in the MeshSmoother module.");
    return;
  }
  else
  {
    error("Unsupported mesh type.  This module only works on Tets and Hexes.");
    return;
  }

//   if (!ifieldhandle->query_scalar_interface(this).get_rep())
//   {
//     error("Input field must contain scalar data.");
//     return;
//   }
  
//   if (ifieldhandle->basis_order() != 1)
//   {
//     error("Isoclipping can only be done for fields with data at nodes.  Note: you can insert a ChangeFieldDataAt module (and an ApplyInterpMatrix module) upstream to push element data to the nodes.");
//     return;
//   }
  cout << "Smoothing hexesb..." << endl;
  
  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = MeshSmootherAlgo::get_compile_info(ftd, ext);
  Handle<MeshSmootherAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile MeshSmoother algorithm.");
    return;
  }

//   FieldHandle ofield = algo->execute(this, ifieldhandle,
// 				     isoval, gui_lte_.get(),
// 				     interp);
  cout << "Smoothing hexesc..." << endl;
  
  FieldHandle ofield = algo->execute(this, ifieldhandle);
    cout << "Smoothing hexesd..." << endl;
  
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Smoothed");
  ofield_port->send_and_dereference(ofield);

//   MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Mapping");
//   omatrix_port->send_and_dereference(interp);
}

CompileInfoHandle
MeshSmootherAlgo::get_compile_info(const TypeDescription *fsrc,
			      string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("MeshSmootherAlgo" + ext);
  static const string base_class_name("MeshSmootherAlgo");

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

