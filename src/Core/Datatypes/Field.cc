/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterfaceAux.h>
#include <Core/Util/ProgressReporter.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Field::type_id("Field", "PropertyManager", 0);


Field::Field()
{
}

Field::~Field()
{
}

const int FIELD_VERSION = 3;

void 
Field::io(Piostream& stream)
{
  int version = stream.begin_class("Field", FIELD_VERSION);
  if (version < 2) {
    // The following was FIELD_VERSION 1 data_at ordering
    //     enum data_location{
    //       NODE,
    //       EDGE,
    //       FACE,
    //       CELL,
    //       NONE
    //     };

    unsigned int tmp;
    int order = 999;
    Pio(stream, tmp);
    if (tmp == 0) {
      // data_at_ was NODE
      order = 1;
      if (mesh()->dimensionality() == 0) order = 0;
    } else if (tmp == 4) {
      // data_at_ was NONE
      order = -1;
    } else {
      // data_at_ was somewhere else
      order = 0;
    }
    
    if (order != basis_order()) {
      // signal error in the stream and return;
      stream.flag_error();
      return;
    }
  } else if (version < 3) {
    int order;
    Pio(stream, order);
    if (order != basis_order()) {
      // signal error in the stream and return;
      stream.flag_error();
      return;
    }
  }

  PropertyManager::io(stream);
  stream.end_class();
}


ScalarFieldInterfaceHandle
Field::query_scalar_interface(ProgressReporter *reporter)
{
  if (basis_order() == -1) { return 0; }

  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = order_type_description();
  CompileInfoHandle ci = ScalarFieldInterfaceMaker::get_compile_info(ftd, ltd);
  LockingHandle<ScalarFieldInterfaceMaker> algo(0);
  
  ProgressReporter my_reporter;
  if (!reporter) reporter = &my_reporter;
  if ( DynamicCompilation::compile( ci, algo, true, reporter ) )
    return algo->make(this);
  else
    return 0;
}


VectorFieldInterfaceHandle
Field::query_vector_interface(ProgressReporter *reporter)
{
  if (basis_order() == -1) { return 0; }
  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = order_type_description();
  CompileInfoHandle ci = VectorFieldInterfaceMaker::get_compile_info(ftd, ltd);
  LockingHandle<VectorFieldInterfaceMaker> algo(0);
  
  ProgressReporter my_reporter;
  if (!reporter) reporter = &my_reporter;
  if ( DynamicCompilation::compile( ci, algo, true, reporter ) )
    return algo->make(this);
  else
    return 0;
}


TensorFieldInterfaceHandle
Field::query_tensor_interface(ProgressReporter *reporter)
{
  if (basis_order() == -1) { return 0; }

  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = order_type_description();
  CompileInfoHandle ci = TensorFieldInterfaceMaker::get_compile_info(ftd, ltd);
  LockingHandle<TensorFieldInterfaceMaker> algo(0);
  
  ProgressReporter my_reporter;
  if (!reporter) reporter = &my_reporter;
  if ( DynamicCompilation::compile( ci, algo, true, reporter ) )
    return algo->make(this);
  else
    return 0;
}

}
