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
 *  DirectInterpolate.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterfaceAux.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

CompileInfoHandle
ScalarFieldInterfaceMaker::get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("SFInterfaceMaker");
  static const string base_class_name("ScalarFieldInterfaceMaker");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ltd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " + ltd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  ltd->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
VectorFieldInterfaceMaker::get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("VFInterfaceMaker");
  static const string base_class_name("VectorFieldInterfaceMaker");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ltd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " + ltd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  ltd->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
TensorFieldInterfaceMaker::get_compile_info(const TypeDescription *ftd,
					    const TypeDescription *ltd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("TFInterfaceMaker");
  static const string base_class_name("TensorFieldInterfaceMaker");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + "." +
		       ltd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() + ", " + ltd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  ltd->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
