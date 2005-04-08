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
 *  Centroids.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/Fields/Centroids.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Core/Util/DynamicCompilation.h>
#include <math.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class Centroids : public Module {
public:
  Centroids(GuiContext* ctx);
  virtual ~Centroids();
  virtual void execute();
};


DECLARE_MAKER(Centroids)


Centroids::Centroids(GuiContext* ctx)
  : Module("Centroids", ctx, Filter, "FieldsCreate", "SCIRun")
{
}


Centroids::~Centroids()
{
}



void
Centroids::execute()
{
  // must find ports and have valid data on inputs
  FieldIPort *ifieldPort = (FieldIPort*)get_iport("TetVolField");
  FieldHandle ifieldhandle;
  if (!ifieldPort->get(ifieldhandle) || !ifieldhandle.get_rep()) return;

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  CompileInfoHandle ci = CentroidsAlgo::get_compile_info(ftd);
  Handle<CentroidsAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, this)) return;

  FieldHandle ofieldhandle(algo->execute(ifieldhandle));
  
  FieldOPort *ofieldPort = (FieldOPort*)get_oport("PointCloudField");
  ofieldPort->send(ofieldhandle);
}



CompileInfoHandle
CentroidsAlgo::get_compile_info(const TypeDescription *field_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("CentroidsAlgoT");
  static const string base_class_name("CentroidsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun


