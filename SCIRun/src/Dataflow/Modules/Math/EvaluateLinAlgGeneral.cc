/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  EvaluateLinAlgGeneral: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Math/EvaluateLinAlgGeneral.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <iostream>

namespace SCIRun {

class EvaluateLinAlgGeneral : public Module
{
private:
  GuiString function_;

public:
  EvaluateLinAlgGeneral(GuiContext* ctx);
  virtual ~EvaluateLinAlgGeneral();
  virtual void execute();
  virtual void presave();
};


DECLARE_MAKER(EvaluateLinAlgGeneral)


EvaluateLinAlgGeneral::EvaluateLinAlgGeneral(GuiContext* ctx)
  : Module("EvaluateLinAlgGeneral", ctx, Filter,"Math", "SCIRun"),
    function_(get_ctx()->subVar("function"), "o1 = i1 * 12;")
{
}


EvaluateLinAlgGeneral::~EvaluateLinAlgGeneral()
{
}


void
EvaluateLinAlgGeneral::execute()
{
  const int mcount = 5;
  MatrixHandle imh[mcount];
  MatrixHandle omh[mcount];
  for (int i = 0; i < mcount; i++)
  {
    imh[i] = 0;
    omh[i] = 0;
  }

  // Get input matrices.
  if (!get_input_handle("i1", imh[0], false))
  {
    remark("i1 is empty.");
  }
  if (!get_input_handle("i2", imh[1], false))
  {
    remark("i2 is empty.");
  }
  if (!get_input_handle("i3", imh[2], false))
  {
    remark("i3 is empty.");
  }
  if (!get_input_handle("i4", imh[3], false))
  {
    remark("i4 is empty.");
  }
  if (!get_input_handle("i5", imh[4], false))
  {
    remark("i5 is empty.");
  }

  int hoffset = 0;
  Handle<EvaluateLinAlgGeneralAlgo> algo;

  // Remove trailing white-space from the function string
  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before get.
  string func=function_.get();
  while (func.size() && isspace(func[func.size()-1]))
    func.resize(func.size()-1);

  for( ;; )
  {
    CompileInfoHandle ci =
      EvaluateLinAlgGeneralAlgo::get_compile_info(mcount, func, hoffset);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Your function would not compile.");
      get_gui()->eval(get_id() + " compile_error "+ci->filename_);
      DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      return;
    }
    if (algo->identify() == func)
    {
      break;
    }
    hoffset++;
  }

  algo->user_function(omh[0], omh[1], omh[2], omh[3], omh[4],
		      imh[0], imh[1], imh[2], imh[3], imh[4]);

  send_output_handle("o1", omh[0]);
  send_output_handle("o2", omh[1]);
  send_output_handle("o3", omh[2]);
  send_output_handle("o4", omh[3]);
  send_output_handle("o5", omh[4]);
}


void
EvaluateLinAlgGeneral::presave()
{
  get_gui()->execute(get_id() + " update_text"); // update gFunction_ before saving.
}



CompileInfoHandle
EvaluateLinAlgGeneralAlgo::get_compile_info(int argc,
				    const string &function,
				    int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("EvaluateLinAlgGeneralInstance" + to_string(hashval)
			     + "_" + to_string(argc));
  static const string base_class_name("EvaluateLinAlgGeneralAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + ".",
		       base_class_name,
		       template_name + ";//",
		       "");

  // Code for the function.
  string class_declaration =
    string("using namespace SCIRun;\n\n") + 
    "class " + template_name + " : public EvaluateLinAlgGeneralAlgo\n" +
    "{\n" +
    "  virtual void user_function(MatrixHandle &o1, MatrixHandle &o2, MatrixHandle &o3, MatrixHandle &o4, MatrixHandle &o5, const MatrixHandle &i1, const MatrixHandle &i2, const MatrixHandle &i3, const MatrixHandle &i4, const MatrixHandle &i5)\n" +
"  {\n" +
    "    " + function + "\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
"  { return string(\"" + string_Cify(function) + "\"); }\n" +
    "};\n";

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_post_include(class_declaration);
  return rval;
}


} // End namespace SCIRun
