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


#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterfaceAux.h>
#include <Core/Util/ProgressReporter.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Field::type_id("Field", "PropertyManager", 0);


Field::Field(data_location at) :
  data_at_(at)
{
}

Field::~Field()
{
}

const int FIELD_VERSION = 1;

void 
Field::io(Piostream& stream){

  stream.begin_class("Field", FIELD_VERSION);
  data_location &tmp = data_at_;
  Pio(stream, (unsigned int&)tmp);
  PropertyManager::io(stream);
  stream.end_class();
}

const string
Field::get_type_name(int n) const
{
  return get_type_description(n)->get_name();
}


ScalarFieldInterfaceHandle
Field::query_scalar_interface(ProgressReporter *reporter)
{
  if (data_at_ == Field::NONE) { return 0; }
  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = data_at_type_description();
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
  if (data_at_ == Field::NONE) { return 0; }
  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = data_at_type_description();
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
  if (data_at_ == Field::NONE) { return 0; }
  const TypeDescription *ftd = get_type_description();
  const TypeDescription *ltd = data_at_type_description();
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
