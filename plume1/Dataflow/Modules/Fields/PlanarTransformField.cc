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
 *  PlanarTransformField.cc:  Rotate and flip field to get it into "standard" view
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
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <Dataflow/Modules/Fields/PlanarTransformField.h>

namespace SCIRun {


class PlanarTransformField : public Module
{
public:
  PlanarTransformField(GuiContext* ctx);
  virtual ~PlanarTransformField();

  virtual void execute();

protected:
  GuiInt Axis_;
  GuiInt TransX_;
  GuiInt TransY_;

  int axis_;
  int tx_;
  int ty_;

  int field_generation_;

  FieldHandle fieldout_;
};


DECLARE_MAKER(PlanarTransformField)

PlanarTransformField::PlanarTransformField(GuiContext* context)
  : Module("PlanarTransformField", context, Filter, "FieldsGeometry", "SCIRun"),
    Axis_(context->subVar("axis")),
    TransX_(context->subVar("trans_x")),
    TransY_(context->subVar("trans_y")),
    axis_(2),
    tx_(0),
    ty_(0),
    field_generation_(-1)
{
}


PlanarTransformField::~PlanarTransformField()
{
}


void
PlanarTransformField::execute()
{
  // Get a handle to the input field port.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");

  FieldHandle fieldin;
  if (!(ifp->get(fieldin) && fieldin.get_rep())) {
    error( "No field handle or representation." );
    return;
  }

  int axis, tx=0, ty=0;

  // Get a handle to the index matrix port.
  MatrixIPort *imp = (MatrixIPort *)get_iport("Index Matrix");

  MatrixHandle matrixin;
  if (imp->get(matrixin) ) {
    if( !matrixin.get_rep()) {
      error( "No index matrix representation." );
      return;
    } else {
      tx = (int) matrixin->get(0, 0);
      ty = (int) matrixin->get(1, 0);
    }
  } else {
    tx   = TransX_.get();
    ty   = TransY_.get();
  }

  axis = Axis_.get();

  if (field_generation_ != fieldin->generation ||
      axis_ != axis ||
      tx_ != tx ||
      ty_ != ty ) {

    field_generation_ = fieldin->generation;

    axis_ = axis;
    tx_ = tx;
    ty_ = ty;

    const TypeDescription *ftd = fieldin->get_type_description();
    CompileInfoHandle ci = PlanarTransformFieldAlgo::get_compile_info(ftd);

    Handle<PlanarTransformFieldAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    fieldout_ = algo->execute(fieldin, axis, tx, ty);
  }
    
  // Get a handle to the output field port.
  if ( fieldout_.get_rep() )
  {
    FieldOPort *ofp = (FieldOPort *)get_oport("Transformed Field");
    ofp->send(fieldout_);
  }
}


CompileInfoHandle
PlanarTransformFieldAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("PlanarTransformFieldAlgoT");
  static const string base_class_name("PlanarTransformFieldAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun

