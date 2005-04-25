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
 *  ClipByFunction.cc:  Clip out parts of a field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/ClipByFunction.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/HashTable.h>
#include <iostream>
#include <stack>

namespace SCIRun {

class ClipByFunction : public Module
{
public:
  ClipByFunction(GuiContext* ctx);
  virtual ~ClipByFunction();

  virtual void execute();

private:
  GuiString gMode_;
  GuiString gFunction_;
  GuiDouble gui_uservar0_;
  GuiDouble gui_uservar1_;
  GuiDouble gui_uservar2_;
  GuiDouble gui_uservar3_;
  GuiDouble gui_uservar4_;
  GuiDouble gui_uservar5_;

  string mode_;
  string function_;

  FieldHandle  fHandle_;
  MatrixHandle mHandle_;

  int fGeneration_;

  bool error_;
};


DECLARE_MAKER(ClipByFunction)

ClipByFunction::ClipByFunction(GuiContext* ctx)
  : Module("ClipByFunction", ctx, Filter, "FieldsCreate", "SCIRun"),
    gMode_(ctx->subVar("clipmode")),
    gFunction_(ctx->subVar("clipfunction")),
    gui_uservar0_(ctx->subVar("u0")),
    gui_uservar1_(ctx->subVar("u1")),
    gui_uservar2_(ctx->subVar("u2")),
    gui_uservar3_(ctx->subVar("u3")),
    gui_uservar4_(ctx->subVar("u4")),
    gui_uservar5_(ctx->subVar("u5")),
    fGeneration_(-1),
    error_(0)
{
}


ClipByFunction::~ClipByFunction()
{
}



void
ClipByFunction::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input");
  FieldHandle fHandle;
  if (!(ifp->get(fHandle) && fHandle.get_rep())) {
    error( "No source field handle or representation" );
    return;
  }
  if (!fHandle->mesh()->is_editable()) {
    error("Not an editable mesh type.");
    error("(Try passing Field through an Unstructure module first).");
    return;
  }

  bool update = false;

  // Check to see if the source field has changed.
  if( fGeneration_ != fHandle->generation ) {
    fGeneration_ = fHandle->generation;
    update = true;
  }

  string mode = gMode_.get();
  string function = gFunction_.get();

  if( mode_     != mode ||
      function_ != function ) {
    update = true;
    
    mode_      = mode;
    function_ = function;
  }

  if( !fHandle_.get_rep() ||
      !mHandle_.get_rep() ||
      update ||
      error_ ) {

    error_ = false;

    // remove trailing white-space from the function string
    while (function.size() && isspace(function[function.size()-1]))
      function.resize(function.size()-1);

    const TypeDescription *ftd = fHandle->get_type_description();
    Handle<ClipByFunctionAlgo> algo;
    int hoffset = 0;

    for (;;) {
      CompileInfoHandle ci =
	ClipByFunctionAlgo::get_compile_info(ftd, function, hoffset);
      if (!DynamicCompilation::compile(ci, algo, false, this)){
	  error("Your function would not compile.");
	  gui->eval(id + " compile_error "+ci->filename_);
	  DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	  error_ = true;
	  return;
	}
      if (algo->identify() == function)
	  break;

      hoffset++;
    }

    int gMode = 0;
    if (gMode_.get() == "cell")
      gMode = 0;
    else if (gMode_.get() == "onenode")
      gMode = 1;
    else if (gMode_.get() == "majoritynodes")
      gMode = 2;
    else if (gMode_.get() == "allnodes")
      gMode = -1;

    // User Variables.
    algo->u0 = gui_uservar0_.get();
    algo->u1 = gui_uservar1_.get();
    algo->u2 = gui_uservar2_.get();
    algo->u3 = gui_uservar3_.get();
    algo->u4 = gui_uservar4_.get();
    algo->u5 = gui_uservar5_.get();

    if (!(fHandle->basis_order() == 0 && gMode == 0 ||
          fHandle->basis_order() == 1 && gMode != 0))
    {
      warning("Basis doesn't match clip location, value will always be zero.");
    }

    fHandle_ = algo->execute(this, fHandle, gMode, mHandle_);
  }


  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *)get_oport("Clipped");
    ofield_port->send(fHandle_);
  }

  if( mHandle_.get_rep() )
  {
    MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Mapping");
    omatrix_port->send(mHandle_);
  }
}


CompileInfoHandle
ClipByFunctionAlgo::get_compile_info(const TypeDescription *fsrc,
				     string clipFunction,
				     int hashoffset)
{
  unsigned int hashval = Hash(clipFunction, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("ClipByFunctionInstance" + to_string(hashval));
  static const string base_class_name("ClipByFunctionAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  string class_declaration =
    string("template <class FIELD>\n") +
    "class " + template_name + " : public ClipByFunctionAlgoT<FIELD>\n" +
    "{\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u0;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u1;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u2;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u3;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u4;\n" +
    "  using ClipByFunctionAlgoT<FIELD>::u5;\n" +
    "\n" +
    "  virtual bool vinside_p(double x, double y, double z,\n" +
    "                         typename FIELD::value_type v)\n" +
    "  {\n" +
    "    return " + clipFunction + ";\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
    "  { return string(\"" + string_Cify(clipFunction) + "\"); }\n" +
    "};\n";

  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  fsrc->fill_compile_info(rval);

  return rval;
}


} // End namespace SCIRun

