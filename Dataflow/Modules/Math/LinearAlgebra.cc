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

#include <iostream>
#include <sci_hash_map.h>

namespace SCIRun {

class LinearAlgebra : public Module
{
private:
  GuiString function_;

public:
  LinearAlgebra(GuiContext* ctx);
  virtual ~LinearAlgebra();
  virtual void execute();
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
  // Get input matrix.
  MatrixIPort *ifp = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrixhandle;
  if (!ifp)
  {
    error("Unable to initialize iport 'Input Matrix'.");
    return;
  }
  if (!(ifp->get(imatrixhandle) && imatrixhandle.get_rep()))
  {
    error("Input matrix is empty.");
    return;
  }

  int hoffset = 0;
  Handle<LinearAlgebraAlgo> algo;
  while (1)
  {
    CompileInfoHandle ci =
      LinearAlgebraAlgo::get_compile_info(function_.get(), hoffset);
    if (!DynamicCompilation::compile(ci, algo, false, this))
    {
      //DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
      error("Your function would not compile.");
      return;
    }
    if (algo->identify() == function_.get())
    {
      break;
    }
    hoffset++;
  }

  MatrixHandle omatrixhandle = algo->function2(imatrixhandle, imatrixhandle);

  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Output Matrix");
  if (!omatrix_port)
  {
    error("Unable to initialize oport 'Output Matrix'.");
    return;
  }

  omatrix_port->send(omatrixhandle);
}


MatrixHandle
LinearAlgebraAlgo::function1(MatrixHandle A)
{
  return 0;
}


MatrixHandle
LinearAlgebraAlgo::function2(MatrixHandle A,
			     MatrixHandle B)
{
  return 0;
}


MatrixHandle
LinearAlgebraAlgo::function3(MatrixHandle A,
			     MatrixHandle B,
			     MatrixHandle C)
{
  return 0;
}


MatrixHandle
LinearAlgebraAlgo::function4(MatrixHandle A,
			     MatrixHandle B,
			     MatrixHandle C,
			     MatrixHandle D)
{
  return 0;
}


MatrixHandle
LinearAlgebraAlgo::function5(MatrixHandle A,
			     MatrixHandle B,
			     MatrixHandle C,
			     MatrixHandle D,
			     MatrixHandle E)
{
  return 0;
}



CompileInfoHandle
LinearAlgebraAlgo::get_compile_info(string function,
				    int hashoffset)

{
  hash<const char *> H;
  unsigned int hashval = H(function.c_str()) + hashoffset;

  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  const string template_name("LinearAlgebraInstance" + to_string(hashval));
  static const string base_class_name("LinearAlgebraAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_name + ".",
		       base_class_name,
		       template_name + ";//",
		       "");

  // Code for the function.
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinearAlgebraAlgo\n" +
"{\n" +
    "  virtual MatrixHandle function2(MatrixHandle A, MatrixHandle B)\n" +
"  {\n" +
    "    return " + function + ";\n" +
    "  }\n" +
    "\n" +
    "  virtual string identify()\n" +
"  { return string(\"" + function + "\"); }\n" +
    "};\n//";

  // Add in the include path to compile this obj
  rval->add_include(include_path + class_declaration);
  return rval;
}


} // End namespace SCIRun
