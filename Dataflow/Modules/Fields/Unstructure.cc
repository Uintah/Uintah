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
#include <Dataflow/Network/Ports/FieldPort.h>
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
  int last_generation_;
  FieldHandle ofieldhandle_;
};


DECLARE_MAKER(Unstructure)
Unstructure::Unstructure(GuiContext* context)
  : Module("Unstructure", context, Filter, "FieldsGeometry", "SCIRun"),
    last_generation_(0)
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

  if (ifieldhandle->generation != last_generation_) {
    update = true;
    last_generation_ = ifieldhandle->generation;
  }

  if( !ofieldhandle_.get_rep() || update )
  {
    string dstname("");
    string dst_basis_name("NoDataBasis");
    const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
    const string &mtdn = mtd->get_name();

    if ((mtdn.find("LatVolMesh") != string::npos) ||
	       (mtdn.find("StructHexVolMesh") != string::npos)) {
      dstname = "HexVolMesh<HexTrilinearLgn<Point> >";
      if (ifieldhandle->basis_order() == 0)
	dst_basis_name = "ConstantBasis";
      else if (ifieldhandle->basis_order() == 1)
	dst_basis_name = "HexTrilinearLgn";
    } else if ((mtdn.find("ImageMesh") != string::npos) ||
	       (mtdn.find("StructQuadSurfMesh") != string::npos)) {
      dstname = "QuadSurfMesh<QuadBilinearLgn<Point> >";
      if (ifieldhandle->basis_order() == 0)
	dst_basis_name = "ConstantBasis";
      else if (ifieldhandle->basis_order() == 1)
	dst_basis_name = "QuadBilinearLgn";
    } else if ((mtdn.find("ScanlineMesh") != string::npos) ||
	       (mtdn.find("StructCurveMesh") != string::npos)) {
      dstname = "CurveMesh<CrvLinearLgn<Point> >";
      if (ifieldhandle->basis_order() == 0)
	dst_basis_name = "ConstantBasis";
      else if (ifieldhandle->basis_order() == 1)
	dst_basis_name = "CrvLinearLgn";
    }

    if (dstname == "") {
      warning("Do not know how to unstructure a " + mtdn + ".");

      ofieldhandle_ = ifieldhandle;
    }
    else {
      const TypeDescription *ftd = ifieldhandle->get_type_description();
      TypeDescription::td_vec *tdv = 
	ifieldhandle->get_type_description(Field::FDATA_TD_E)->get_sub_type();
      string data_name = (*tdv)[0]->get_name();
      
      CompileInfoHandle ci = UnstructureAlgo::get_compile_info(ftd, dstname, 
							       dst_basis_name,
							       data_name);
      Handle<UnstructureAlgo> algo;

      if (!module_dynamic_compile(ci, algo)) return;

      ofieldhandle_ = algo->execute(this, ifieldhandle);

      if (ofieldhandle_.get_rep())
	ofieldhandle_->copy_properties(ifieldhandle.get_rep());
    }
  }

  send_output_handle("Output Field", ofieldhandle_, true);
}



CompileInfoHandle
UnstructureAlgo::get_compile_info(const TypeDescription *fsrc,
				  const string &mesh_dst,
				  const string &basis_dst,
				  const string &data_dst)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("UnstructureAlgoT");
  static const string base_class_name("UnstructureAlgo");

  const string fdstname = "GenericField<" + mesh_dst + ", " + 
    basis_dst + "<" + data_dst + ">, vector<" + data_dst + "> > ";

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + "." +
		       to_filename(fdstname) + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name() + "," + fdstname);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  rval->add_mesh_include("../src/Core/Datatypes/CurveMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/QuadSurfMesh.h");
  rval->add_mesh_include("../src/Core/Datatypes/HexVolMesh.h");

  fsrc->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
