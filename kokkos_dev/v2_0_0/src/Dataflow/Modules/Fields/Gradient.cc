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
  FieldHandle fieldout_;

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

  if (!ifp)
  {
    error( "Unable to initialize iport 'Input Field'.");
    return;
  }

  if (!(ifp->get(fieldin) && fieldin.get_rep()))
  {
    error( "No handle or representation in input field." );
    return;
  }

  if (!fieldin->query_scalar_interface(this).get_rep())
  {
    error( "This module only works on fields of scalar data." );
    return;
  }

  // If no data or a changed recalcute.
  if( !fieldout_.get_rep() ||
      fGeneration_ != fieldin->generation )
  {
    fGeneration_ = fieldin->generation;

    const TypeDescription *ftd  = fieldin->get_type_description(0);
    const TypeDescription *ttd = fieldin->get_type_description(1);

    CompileInfoHandle ci = GradientAlgo::get_compile_info(ftd,ttd);
    Handle<GradientAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fieldout_ = algo->execute(fieldin);
  }

  // Get a handle to the output field port.
  if ( fieldout_.get_rep() )
  {
    FieldOPort* ofp = (FieldOPort *) get_oport("Output Gradient");

    if (!ofp)
    {
      error("Unable to initialize oport 'Output Gradient'.");
      return;
    }

    // Send the data downstream
    ofp->send(fieldout_);
  }
}

CompileInfoHandle
GradientAlgo::get_compile_info(const TypeDescription *ftd,
			       const TypeDescription *ttd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("GradientAlgoT");
  static const string base_class_name("GradientAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() +
		       ttd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 

#ifdef __sgi
                       ftd->get_name() + "<" + ttd->get_name()+ " >" + ", " +
                       ftd->get_name() + "<Vector > ");
#else
                       ftd->get_name() + ", " +
		       ttd->get_name());
#endif
  
  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun



