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
  MatrixHandle imh[mcount];
  MatrixHandle omh[mcount];
  for (int i = 0; i < mcount; i++)
  {
    imh[i] = 0;
    omh[i] = 0;
  }

  // Get input matrices.
  MatrixIPort *ifp0 = (MatrixIPort *)get_iport("i1");
  if (!ifp0)
  {
    error("Unable to initialize iport 'i1'.");
    return;
  }
  if (!(ifp0->get(imh[0]) && imh[0].get_rep()))
  {
    remark("i1 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp1 = (MatrixIPort *)get_iport("i2");
  if (!ifp1)
  {
    error("Unable to initialize iport 'i2'.");
    return;
  }
  if (!(ifp1->get(imh[1]) && imh[1].get_rep()))
  {
    remark("i2 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp2 = (MatrixIPort *)get_iport("i3");
  if (!ifp2)
  {
    error("Unable to initialize iport 'i3'.");
    return;
  }
  if (!(ifp2->get(imh[2]) && imh[2].get_rep()))
  {
    remark("i3 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp3 = (MatrixIPort *)get_iport("i4");
  if (!ifp3)
  {
    error("Unable to initialize iport 'i4'.");
    return;
  }
  if (!(ifp3->get(imh[3]) && imh[3].get_rep()))
  {
    remark("i4 is empty.");
  }

  // Get input matrices.
  MatrixIPort *ifp4 = (MatrixIPort *)get_iport("i5");
  if (!ifp4)
  {
    error("Unable to initialize iport 'i5'.");
    return;
  }
  if (!(ifp4->get(imh[4]) && imh[4].get_rep()))
  {
    remark("i5 is empty.");
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

  algo->user_function(omh[0], omh[1], omh[2], omh[3], omh[4],
		      imh[0], imh[1], imh[2], imh[3], imh[4]);

  MatrixOPort *omatrix_port1 = (MatrixOPort *)get_oport("o1");
  if (!omatrix_port1)
  {
    error("Unable to initialize oport 'o1'.");
    return;
  }
  omatrix_port1->send(omh[0]);

  MatrixOPort *omatrix_port2 = (MatrixOPort *)get_oport("o2");
  if (!omatrix_port2)
  {
    error("Unable to initialize oport 'o2'.");
    return;
  }
  omatrix_port2->send(omh[1]);

  MatrixOPort *omatrix_port3 = (MatrixOPort *)get_oport("o3");
  if (!omatrix_port3)
  {
    error("Unable to initialize oport 'o3'.");
    return;
  }
  omatrix_port3->send(omh[2]);

  MatrixOPort *omatrix_port4 = (MatrixOPort *)get_oport("o4");
  if (!omatrix_port4)
  {
    error("Unable to initialize oport 'o4'.");
    return;
  }
  omatrix_port4->send(omh[3]);

  MatrixOPort *omatrix_port5 = (MatrixOPort *)get_oport("o5");
  if (!omatrix_port5)
  {
    error("Unable to initialize oport 'o5'.");
    return;
  }
  omatrix_port5->send(omh[4]);
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

  // Code for the function.
  string class_declaration =
    string("\"\n\nusing namespace SCIRun;\n\n") + 
    "class " + template_name + " : public LinearAlgebraAlgo\n" +
"{\n" +
    "  virtual void user_function(MatrixHandle &o1, MatrixHandle &o2, MatrixHandle &o3, MatrixHandle &o4, MatrixHandle &o5, const MatrixHandle &i1, const MatrixHandle &i2, const MatrixHandle &i3, const MatrixHandle &i4, const MatrixHandle &i5)\n" +
"  {\n" +
    "    " + function + "\n" +
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
