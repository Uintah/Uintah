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
  const int mcount = 5;
  MatrixHandle imatrixhandle[mcount];
  for (int i = 0; i < mcount; i++)
  {
    imatrixhandle[i] = 0;
  }

  // Get input matrices.
  MatrixIPort *ifp0 = (MatrixIPort *)get_iport("A");
  if (!ifp0)
  {
    error("Unable to initialize iport 'A'.");
    return;
  }
  if (!(ifp0->get(imatrixhandle[0]) && imatrixhandle[0].get_rep()))
  {
  }

  // Get input matrices.
  MatrixIPort *ifp1 = (MatrixIPort *)get_iport("B");
  if (!ifp1)
  {
    error("Unable to initialize iport 'B'.");
    return;
  }
  if (!(ifp1->get(imatrixhandle[1]) && imatrixhandle[1].get_rep()))
  {
  }

  // Get input matrices.
  MatrixIPort *ifp2 = (MatrixIPort *)get_iport("C");
  if (!ifp2)
  {
    error("Unable to initialize iport 'C'.");
    return;
  }
  if (!(ifp2->get(imatrixhandle[2]) && imatrixhandle[2].get_rep()))
  {
  }

  // Get input matrices.
  MatrixIPort *ifp3 = (MatrixIPort *)get_iport("D");
  if (!ifp3)
  {
    error("Unable to initialize iport 'D'.");
    return;
  }
  if (!(ifp3->get(imatrixhandle[3]) && imatrixhandle[3].get_rep()))
  {
  }

  // Get input matrices.
  MatrixIPort *ifp4 = (MatrixIPort *)get_iport("E");
  if (!ifp4)
  {
    error("Unable to initialize iport 'E'.");
    return;
  }
  if (!(ifp4->get(imatrixhandle[4]) && imatrixhandle[4].get_rep()))
  {
  }

  int hoffset = 0;
  Handle<LinearAlgebraAlgo> algo;
  while (1)
  {
    CompileInfoHandle ci =
      LinearAlgebraAlgo::get_compile_info(mcount, function_.get(), hoffset);
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

  MatrixHandle omatrixhandle(0);
  switch(mcount)
  {
  case 0:
    omatrixhandle = algo->function0();
    break;
  case 1:
    omatrixhandle = algo->function1(imatrixhandle[0]);
    break;
  case 2:
    omatrixhandle = algo->function2(imatrixhandle[0],
				    imatrixhandle[1]);
    break;
  case 3:
    omatrixhandle = algo->function3(imatrixhandle[0],
				    imatrixhandle[1],
				    imatrixhandle[2]);
    break;
  case 4:
    omatrixhandle = algo->function4(imatrixhandle[0],
				    imatrixhandle[1],
				    imatrixhandle[2],
				    imatrixhandle[3]);
    break;
  case 5:
    omatrixhandle = algo->function5(imatrixhandle[0],
				    imatrixhandle[1],
				    imatrixhandle[2],
				    imatrixhandle[3],
				    imatrixhandle[4]);
    break;
  default:
    ; // some error
  }

  MatrixOPort *omatrix_port = (MatrixOPort *)get_oport("Output Matrix");
  if (!omatrix_port)
  {
    error("Unable to initialize oport 'Output Matrix'.");
    return;
  }

  omatrix_port->send(omatrixhandle);
}


MatrixHandle
LinearAlgebraAlgo::function0()
{
  return 0;
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
LinearAlgebraAlgo::get_compile_info(int argc,
				    string function,
				    int hashoffset)

{
  hash<const char *> H;
  unsigned int hashval = H(function.c_str()) + hashoffset;

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

  string prototype;
  switch(argc)
  {
  case 0:
    prototype = "function0()";
    break;
  case 1:
    prototype = "function1(MatrixHandle A)";
    break;
  case 2:
    prototype = "function2(MatrixHandle A, MatrixHandle B)";
    break;
  case 3:
    prototype = "function3(MatrixHandle A, MatrixHandle B, MatrixHandle C)";
    break;
  case 4:
    prototype = "function4(MatrixHandle A, MatrixHandle B, MatrixHandle C, MatrixHandle D)";
    break;
  case 5:
    prototype = "function5(MatrixHandle A, MatrixHandle B, MatrixHandle C, MatrixHandle D, MatrixHandle E)";
    break;
  default:
    ; // error
  }

  // Code for the function.
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinearAlgebraAlgo\n" +
"{\n" +
    "  virtual MatrixHandle " + prototype + "\n" +
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
