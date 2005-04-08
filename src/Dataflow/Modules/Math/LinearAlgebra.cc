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
 *  LinearAlgebra: Unary field data operations
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
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Modules/Math/LinearAlgebra.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <iostream>

namespace SCIRun {

class LinearAlgebra : public Module
{
private:
  GuiString function_;

public:
  LinearAlgebra(GuiContext* ctx);
  virtual ~LinearAlgebra();
  virtual void execute();
  virtual void presave();
};


DECLARE_MAKER(LinearAlgebra)


LinearAlgebra::LinearAlgebra(GuiContext* ctx)
  : Module("LinearAlgebra", ctx, Filter,"Math", "SCIRun"),
    function_(ctx->subVar("function"))
{
}


LinearAlgebra::~LinearAlgebra()
{
}


void
LinearAlgebra::execute()
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
  MatrixIPort *ifp0 = (MatrixIPort *)get_iport("i1");
  if (!(ifp0->get(imh[0]) && imh[0].get_rep()))
  {
    remark("i1 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp1 = (MatrixIPort *)get_iport("i2");
  if (!(ifp1->get(imh[1]) && imh[1].get_rep()))
  {
    remark("i2 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp2 = (MatrixIPort *)get_iport("i3");
  if (!(ifp2->get(imh[2]) && imh[2].get_rep()))
  {
    remark("i3 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp3 = (MatrixIPort *)get_iport("i4");
  if (!(ifp3->get(imh[3]) && imh[3].get_rep()))
  {
    remark("i4 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp4 = (MatrixIPort *)get_iport("i5");
  if (!(ifp4->get(imh[4]) && imh[4].get_rep()))
  {
    remark("i5 is empty.");
  }

  int hoffset = 0;
  Handle<LinearAlgebraAlgo> algo;

  // Remove trailing white-space from the function string
  gui->execute(id + " update_text"); // update gFunction_ before get.
  string func=function_.get();
  while (func.size() && isspace(func[func.size()-1]))
    func.resize(func.size()-1);

  for( ;; )
  {
    CompileInfoHandle ci =
      LinearAlgebraAlgo::get_compile_info(mcount, func, hoffset);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      error("Your function would not compile.");
      gui->eval(id + " compile_error "+ci->filename_);
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

  MatrixOPort *omatrix_port1 = (MatrixOPort *)get_oport("o1");
  omatrix_port1->send(omh[0]);

  MatrixOPort *omatrix_port2 = (MatrixOPort *)get_oport("o2");
  omatrix_port2->send(omh[1]);

  MatrixOPort *omatrix_port3 = (MatrixOPort *)get_oport("o3");
  omatrix_port3->send(omh[2]);

  MatrixOPort *omatrix_port4 = (MatrixOPort *)get_oport("o4");
  omatrix_port4->send(omh[3]);

  MatrixOPort *omatrix_port5 = (MatrixOPort *)get_oport("o5");
  omatrix_port5->send(omh[4]);
}


void
LinearAlgebra::presave()
{
  gui->execute(id + " update_text"); // update gFunction_ before saving.
}



CompileInfoHandle
LinearAlgebraAlgo::get_compile_info(int argc,
				    const string &function,
				    int hashoffset)

{
  unsigned int hashval = Hash(function, 0x7fffffff) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("LinearAlgebraInstance" + to_string(hashval)
			     + "_" + to_string(argc));
  static const string base_class_name("LinearAlgebraAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + ".",
		       base_class_name,
		       template_name + ";//",
		       "");

  // Code for the function.
  string class_declaration =
    string("using namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinearAlgebraAlgo\n" +
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
