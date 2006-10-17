//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : InterfaceWithTetGen.cc
//    Author : Martin Cole
//    Date   : Wed Mar 22 07:56:22 2006

#include <Core/Thread/Mutex.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/Field.h>
#include <Core/Basis/NoData.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/InterfaceWithTetGen.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

namespace SCIRun {

using std::cerr;
using std::endl;

class InterfaceWithTetGen : public Module
{
public:
  InterfaceWithTetGen(GuiContext* ctx);
  virtual ~InterfaceWithTetGen();

  virtual void execute();
private:
  GuiString switch_;
};

DECLARE_MAKER(InterfaceWithTetGen)
  
InterfaceWithTetGen::InterfaceWithTetGen(GuiContext* ctx) : 
  Module("InterfaceWithTetGen", ctx, Filter, "NewField", "SCIRun"),
  switch_(ctx->subVar("switch"), "pqYAz")
{ 
}


InterfaceWithTetGen::~InterfaceWithTetGen()
{
}


void
InterfaceWithTetGen::execute() 
{
  tetgenio in, out;

  FieldHandle main_input;
  if(!get_input_handle( "Main", main_input, true)) return;

  FieldHandle points;
  if (get_input_handle( "Points", points, false)) {
    // Process the extra interior points.
    error("add the interior points");
  }

  FieldHandle region_attribs;
  if (get_input_handle( "Region Attribs", region_attribs, false)) 
  {
    const TypeDescription *ratd = region_attribs->get_type_description();
    const string tcn("TGRegionAttrib");
    CompileInfoHandle ci = TGRegionAttribAlgo::get_compile_info(ratd, tcn);
    Handle<TGRegionAttribAlgo> raalgo;
    if (module_dynamic_compile(ci, raalgo)) {
      // Process the region attributes.
      raalgo->set_region_attribs(this, region_attribs, in);
    }
  }

  unsigned idx = 0;
  unsigned fidx = 0;

  // indices start from 0.
  in.firstnumber = 0;
  in.mesh_dim = 3;

  
  //! Add the info for the outer surface, or tetvol to be refined.
  const TypeDescription *ostd = main_input->get_type_description();
  const string tcn("TGSurfaceTGIO");
  CompileInfoHandle ci = TGSurfaceTGIOAlgo::get_compile_info(ostd, tcn);
  Handle<TGSurfaceTGIOAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) {
    error("Could not get InterfaceWithTetGen/SCIRun converter algorithm");
    return;
  }
  int marker = -10;
  algo->to_tetgenio(this, main_input, idx, fidx, marker, in);

  // Interior surfaces -- each a new boundary.
  vector<FieldHandle> interior_surfaces;
  if (get_dynamic_input_handles("Regions", interior_surfaces, false)) {
    // Add each interior surface to the tetgenio input class.
    vector<FieldHandle>::iterator iter = interior_surfaces.begin();
    while (iter != interior_surfaces.end()) {
      FieldHandle insurf = *iter++;
      const TypeDescription *rtd = insurf->get_type_description();
      CompileInfoHandle ci = TGSurfaceTGIOAlgo::get_compile_info(rtd, tcn);
      Handle<TGSurfaceTGIOAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) {
	error("Could not get InterfaceWithTetGen/SCIRun surface input algorithm");
	return;
      }    
      marker *= 2;
      algo->to_tetgenio(this, insurf, idx, fidx, marker, in);
    }
  }
  update_progress(.2);
  // Save files for later debugging.
  in.save_nodes("/tmp/tgIN");
  in.save_poly("/tmp/tgIN");

  // Create the new mesh.
  tetrahedralize((char*)switch_.get().c_str(), &in, &out); 
  FieldHandle tetvol_out;
    update_progress(.9);
  // Convert to a SCIRun TetVol.
  tetvol_out = algo->to_tetvol(out);
  update_progress(1.0);
  if (tetvol_out.get_rep() == 0) { return; }
  send_output_handle("TetVol", tetvol_out);
}

CompileInfoHandle
InterfaceWithTetGenInterface::get_compile_info(const TypeDescription *td, 
				  const string template_class_name)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("InterfaceWithTetGenInterface");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       td->get_name(".", ".") + ".",
                       base_class_name, 
                       template_class_name, 
                       td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  td->fill_compile_info(rval);
  rval->add_basis_include("../src/Core/Basis/Constant.h");
  rval->add_basis_include("../src/Core/Basis/TetLinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/TetVolMesh.h");
  return rval;
}


} // end namespace SCIRun

