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
 *  DirectInterpolateAlgo.cc:
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Datatypes/Field.h>
#include <Dataflow/Modules/Fields/DirectInterpolateAlgo.h>

namespace SCIRun {


// This should go in the .cc file.
DirectInterpAlgoBase::~DirectInterpAlgoBase()
{}


static string
strip(const string s)
{
  string result;
  for (unsigned int i = 0; i < s.size(); i++)
  {
    if (isalnum(s[i]))
    {
      result += s[i];
    }
  }
  return result;
}

CompileInfo *
DirectInterpAlgoBase::get_compile_info(const TypeDescription *td0,
				       const TypeDescription *td1)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("DirectInterpAlgo");
  static const string base_class_name("DirectInterpAlgoBase");

  CompileInfo *rval = 
    scinew CompileInfo(strip(template_class_name + "." + td0->get_name(".", ".") +
		       td1->get_name(".", ".")) + ".",
                       base_class_name, 
                       template_class_name, 
                       td0->get_name() + ", " + td1->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  td0->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
