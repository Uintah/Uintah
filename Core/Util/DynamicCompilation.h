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
  Moduleions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  DynamicCompilation.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 */


#ifndef SCIRun_Core_Util_DynamicCompilation_h
#define SCIRun_Core_Util_DynamicCompilation_h

#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {


class SCICORESHARE DynamicCompilation
{
public:
  virtual ~DynamicCompilation();

  template <class DC> static bool compile(CompileInfoHandle ci, DC &result);
  template <class DC> static bool compile(CompileInfoHandle ci, DC &result, bool ignore);
  template <class DC> static bool compile(CompileInfoHandle ci, DC &, ProgressReporter *);
  template <class DC> static bool compile(CompileInfoHandle ci, DC &, bool , ProgressReporter *);
};



template <class DC>
bool
DynamicCompilation::compile(CompileInfoHandle cih, DC &result)
{
  ProgressReporter reporter;
  return compile( cih, result, false, &reporter );
}

template <class DC>
bool
DynamicCompilation::compile(CompileInfoHandle cih, DC &result, bool ignore)
{
  ProgressReporter reporter;
  return compile( cih, result, ignore, &reporter );
}

template <class DC>
bool
DynamicCompilation::compile(CompileInfoHandle cih, DC &result, 
			      ProgressReporter *reporter)
{
  return compile( cih, result, false, reporter );
}

template <class DC>
bool
  DynamicCompilation::compile(CompileInfoHandle cih, DC &result, 
			      bool ignore, 
			      ProgressReporter *reporter)
{
  ASSERT(cih.get_rep());
  const CompileInfo &ci = *(cih.get_rep());
  DynamicAlgoHandle algo_handle;
  bool status = false;

  reporter->report_progress( ProgressReporter::Starting );
  
  if (! DynamicLoader::scirun_loader().fetch(ci, algo_handle))
  {
    reporter->report_progress( ProgressReporter::Compiling);

    status = DynamicLoader::scirun_loader().compile_and_store(ci, 
							      ignore, 
							      reporter->msgStream());

    reporter->report_progress( ProgressReporter::CompilationDone );

    if (! (status && DynamicLoader::scirun_loader().fetch(ci, algo_handle)))
    {
      reporter->error("Could not compile algorithm for '" +
	    ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
      return false;
    }
  }

  result = dynamic_cast<typename DC::pointer_type>(algo_handle.get_rep());
  if (result.get_rep() == 0) 
  {
    if ( !ignore ) reporter->error("Could not get algorithm for '" +
				   ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
    status = false;
  }
  else
    status = true;

  reporter->report_progress( ProgressReporter::Done );

  return status;
}

} // Namespace SCIRun

#endif
