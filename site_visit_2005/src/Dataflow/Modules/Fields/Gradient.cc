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
 *  Gradient.cc:  Unfinished modules
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
#include <Core/Containers/Handle.h>

#include <Dataflow/Modules/Fields/Gradient.h>

namespace SCIRun {

class Gradient : public Module
{
public:
  Gradient(GuiContext* ctx);
  virtual ~Gradient();

  virtual void execute();

protected:
  int fGeneration_;
};


DECLARE_MAKER(Gradient)

Gradient::Gradient(GuiContext* ctx)
  : Module("Gradient", ctx, Filter, "FieldsData", "SCIRun"),
    fGeneration_(-1)
{
}

Gradient::~Gradient()
{
}

void
Gradient::execute()
{
  FieldIPort* ifp = (FieldIPort *)get_iport("Input Field");

  FieldHandle fieldin;
  if (!(ifp->get(fieldin) && fieldin.get_rep()))
  {
    error( "No handle or representation in input field." );
    return;
  }

  TypeDescription *otd = 0;

  if (fieldin->query_scalar_interface(this).get_rep() )
  {
    otd = (TypeDescription *) SCIRun::get_type_description( (Vector*) 0 );
  }
  else if (fieldin->query_vector_interface(this).get_rep())
  {
    otd = (TypeDescription *) SCIRun::get_type_description( (Tensor*) 0 );
  }
  else
  {
    error( "This module only works on fields of scalar or vector data.");
    return;
  }

  // If no data or a changed recalcute.
  FieldHandle fieldout(0);
  if( fGeneration_ != fieldin->generation )
  {
    fGeneration_ = fieldin->generation;

    const TypeDescription *ftd = fieldin->get_type_description(0);
    const TypeDescription *ttd = fieldin->get_type_description(1);

    CompileInfoHandle ci =
      GradientAlgo::get_compile_info(ftd, ttd, otd);
    Handle<GradientAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fieldout = algo->execute(fieldin);
  }

  // Send the data downstream
  if ( fieldout.get_rep() )
  {
    FieldOPort* ofp = (FieldOPort *) get_oport("Output Gradient");
    ofp->send(fieldout);
  }
}

CompileInfoHandle
GradientAlgo::get_compile_info(const TypeDescription *ftd,
			       const TypeDescription *ttd,
			       const TypeDescription *otd )
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GradientAlgoT");
  static const string base_class_name("GradientAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + "<" + ttd->get_name() + "> " + ", " +
                       ftd->get_name() + "<" + otd->get_name() + "> " );
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
