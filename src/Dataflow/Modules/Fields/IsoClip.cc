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
 *  IsoClip.cc:  Clip out parts of a field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/IsoClip.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {

int IsoClipAlgo::tet_permute_table[15][4] = {
  { 0, 0, 0, 0 }, // 0x0
  { 3, 0, 2, 1 }, // 0x1
  { 2, 3, 0, 1 }, // 0x2
  { 0, 1, 2, 3 }, // 0x3
  { 1, 2, 0, 3 }, // 0x4
  { 0, 2, 3, 1 }, // 0x5
  { 0, 3, 1, 2 }, // 0x6
  { 0, 1, 2, 3 }, // 0x7
  { 0, 1, 2, 3 }, // 0x8
  { 2, 1, 3, 0 }, // 0x9
  { 3, 1, 0, 2 }, // 0xa
  { 1, 2, 0, 3 }, // 0xb
  { 3, 2, 1, 0 }, // 0xc
  { 2, 3, 0, 1 }, // 0xd
  { 3, 0, 2, 1 }, // 0xe
};


int IsoClipAlgo::tri_permute_table[7][3] = {
  { 0, 0, 0 }, // 0x0
  { 2, 0, 1 }, // 0x1
  { 1, 2, 0 }, // 0x2
  { 0, 1, 2 }, // 0x3
  { 0, 1, 2 }, // 0x4
  { 1, 2, 0 }, // 0x5
  { 2, 0, 1 }, // 0x6
};

class IsoClip : public Module
{
private:
  GuiDouble gui_isoval_;
  GuiInt    gui_lte_;
  int       last_field_generation_;
  double    last_isoval_;
  int       last_lte_;
  int       last_matrix_generation_;

public:
  IsoClip(GuiContext* ctx);
  virtual ~IsoClip();

  virtual void execute();
};


DECLARE_MAKER(IsoClip)


IsoClip::IsoClip(GuiContext* ctx)
  : Module("IsoClip", ctx, Filter, "Fields", "SCIRun"),
    gui_isoval_(ctx->subVar("isoval")),
    gui_lte_(ctx->subVar("lte")),
    last_field_generation_(0),
    last_isoval_(0),
    last_lte_(-1),
    last_matrix_generation_(0)
{
}


IsoClip::~IsoClip()
{
}


void
IsoClip::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  FieldHandle ifieldhandle;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }

  MatrixIPort *imp = (MatrixIPort *)get_iport("Optional Isovalue");
  if (!imp)
  {
    error("Unable to initialize iport 'Optional Isovalue'.");
    return;
  }
  MatrixHandle isomat;
  if (imp->get(isomat) && isomat.get_rep() &&
      isomat->nrows() > 0 && isomat->ncols() > 0 &&
      isomat->generation != last_matrix_generation_)
  {
    last_matrix_generation_ = isomat->generation;
    gui_isoval_.set(isomat->get(0, 0));
  }

  const double isoval = gui_isoval_.get();
  if (last_field_generation_ == ifieldhandle->generation &&
      last_isoval_ == isoval &&
      last_lte_ == gui_lte_.get())
  {
    // We're up to date, return.
    return;
  }
  last_field_generation_ = ifieldhandle->generation;
  last_isoval_ = isoval;
  last_lte_ = gui_lte_.get();

  string ext = "";
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  if (mtd->get_name() == "TetVolMesh")
  {
    ext = "Tet";
  }
  else if (mtd->get_name() == "TriSurfMesh")
  {
    ext = "Tri";
  }
  else
  {
    error("Unsupported mesh type.  This module only works on Tets and Tris.");
    return;
  }

  if (!ifieldhandle->query_scalar_interface(this).get_rep())
  {
    error("Input field must contain scalar data.");
    return;
  }
  
  if (ifieldhandle->data_at() != Field::NODE)
  {
    error("Isoclipping can only done for fields with data at nodes.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = IsoClipAlgo::get_compile_info(ftd, ext);
  Handle<IsoClipAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Unable to compile IsoClip algorithm.");
    return;
  }

  FieldHandle ofield = algo->execute(this, ifieldhandle,
				     isoval, gui_lte_.get());
  
  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  if (!ofield_port)
  {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  ofield_port->send(ofield);
}



CompileInfoHandle
IsoClipAlgo::get_compile_info(const TypeDescription *fsrc,
			      string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("IsoClipAlgo" + ext);
  static const string base_class_name("IsoClipAlgo");

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

