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
 *  QuadToTri.cc:  Convert a Quad field into a Tri field using 1-5 split
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Dataflow/Modules/Fields/QuadToTri.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <vector>
#include <algorithm>


namespace SCIRun {

class QuadToTri : public Module {
private:
  int last_generation_;
  FieldHandle ofieldhandle;

public:

  //! Constructor/Destructor
  QuadToTri(GuiContext *context);
  virtual ~QuadToTri();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(QuadToTri)


QuadToTri::QuadToTri(GuiContext *context) : 
  Module("QuadToTri", context, Filter, "FieldsGeometry", "SCIRun"),
  last_generation_(0)
{
}

QuadToTri::~QuadToTri()
{
}

void
QuadToTri::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("QuadSurf");
  FieldHandle ifieldhandle;
  if(!(ifieldport->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Can't get field.");
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("TriSurf");

  // Cache generation.
  if (ifieldhandle->generation == last_generation_)
  {
    ofp->send(ofieldhandle);
    return;
  }
  last_generation_ = ifieldhandle->generation;
  
  const TypeDescription *src_td = ifieldhandle->get_type_description();

  const string iname =
    ifieldhandle->mesh()->get_type_description()->get_name();
  if (iname == "ImageMesh" || iname == "StructQuadSurfMesh")
  {
    CompileInfoHandle ici = ImgToTriAlgo::get_compile_info(src_td);
    Handle<ImgToTriAlgo> ialgo;
    if (DynamicCompilation::compile(ici, ialgo, true, this))
    {
      if (!ialgo->execute(ifieldhandle, ofieldhandle, this))
      {
	warning("ImgToTri conversion failed to copy data.");
	return;
      }
    }
    else
    {
      error("QuadToTri only supports Quad field types.");
      return;
    }
  }
  else
  {
    CompileInfoHandle qci = QuadToTriAlgo::get_compile_info(src_td);
    Handle<QuadToTriAlgo> qalgo;
    if (DynamicCompilation::compile(qci, qalgo, true, this))
    {
      if (!qalgo->execute(ifieldhandle, ofieldhandle, this))
      {
	warning("QuadToTri conversion failed to copy data.");
	return;
      }
    }
    else
    {
      error("QuadToTri only supports Quad field types.");
      return;
    }
  }
  ofp->send(ofieldhandle);
}

CompileInfoHandle
QuadToTriAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("QuadToTriAlgoT");
  static const string base_class_name("QuadToTriAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       src_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       src_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  src_td->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
ImgToTriAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ImgToTriAlgoT");
  static const string base_class_name("ImgToTriAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       src_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       src_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  src_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
