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
 *  MoveElemToNode.cc:  Rotate and flip field to get it into "standard" view
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
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/MoveElemToNode.h>
#include <iostream>

namespace SCIRun {


class MoveElemToNode : public Module
{
public:
  MoveElemToNode(GuiContext* ctx);
  virtual ~MoveElemToNode();

  virtual void execute();

protected:
  int ifield_generation_;
};


DECLARE_MAKER(MoveElemToNode)

MoveElemToNode::MoveElemToNode(GuiContext* ctx)
  : Module("MoveElemToNode", ctx, Filter, "FieldsData", "SCIRun"),
    ifield_generation_(0)
{
}


MoveElemToNode::~MoveElemToNode()
{
}


void
MoveElemToNode::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Elem Field");
  FieldHandle ifield;
  if (!(ifp->get(ifield) && ifield.get_rep()))
  {
    return;
  }

  // Get the output port now, because we may be able to pass the field
  // directly through if it is already cell centered.
  FieldOPort *ofp = (FieldOPort *)get_oport("Node Field");

  string ext = "";
  const TypeDescription *mtd = ifield->mesh()->get_type_description();
  if (mtd->get_name() == "LatVolMesh")
  {
    if (ifield->basis_order() != 0)
    {
      remark("Field is already cell centered.  Passing through.");
      ofp->send(ifield);
      return;
    }
    ext = "Lat";
  }
  else if (mtd->get_name() == "StructHexVolMesh")
  {
    if (ifield->basis_order() != 0)
    {
      remark("Field is already cell centered.  Passing through.");
      ofp->send(ifield);
      return;
    }
    ext = "SHex";
  }
  else if (mtd->get_name() == "ImageMesh")
  {
    ext = "Img";
  }
  else if (mtd->get_name() == "StructQuadSurfMesh")
  {
    ext = "SQuad";
  }
  else
  {
    error("Unsupported mesh type.  This module only works on LatVols, StructHexVols, Images, and StructQuadSurfs.");
    return;
  }

  if (ifield_generation_ != ifield->generation)
  {
    const TypeDescription *ftd = ifield->get_type_description();
    CompileInfoHandle ci = MoveElemToNodeAlgo::get_compile_info(ftd, ext);
    Handle<MoveElemToNodeAlgo> algo;
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Unable to compile MoveElemToNode algorithm.");
      return;
    }

    FieldHandle ofield = algo->execute(this, ifield);
  
    ofp->send(ofield);
  }
}


CompileInfoHandle
MoveElemToNodeAlgo::get_compile_info(const TypeDescription *fsrc, string ext)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_class_name("MoveElemToNodeAlgo" + ext);
  static const string base_class_name("MoveElemToNodeAlgo");

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

