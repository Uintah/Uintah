/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
 */


#ifndef SCIRun_Core_Util_DynamicCompilation_h
#define SCIRun_Core_Util_DynamicCompilation_h

#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <iostream>

namespace SCIRun {


class DynamicCompilation
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
  bool did_compile = false;

  reporter->report_progress( ProgressReporter::Starting );
  
  if (! DynamicLoader::scirun_loader().fetch(ci, algo_handle))
  {
    reporter->report_progress( ProgressReporter::Compiling);

    status =
      DynamicLoader::scirun_loader().compile_and_store(ci, ignore, reporter);

    did_compile = true;
    reporter->report_progress( ProgressReporter::CompilationDone );

    if (! (status && DynamicLoader::scirun_loader().fetch(ci, algo_handle)))
    {
      reporter->error("Could not compile algorithm for '" +
                      ci.template_class_name_ + "<" +
                      ci.template_arg_ + ">'.");
      return false;
    }
  }

  result = dynamic_cast<typename DC::pointer_type>(algo_handle.get_rep());
  if (result.get_rep() == 0) 
  {
    if ( !ignore )
    {
      reporter->error("Could not get algorithm for '" +
                      ci.template_class_name_ + "<" +
                      ci.template_arg_ + ">'.");
    }
    status = false;
  }
  else
  {
    //! We have the maker, now destroy the library file if we do not
    //! need it anymore.

    if ((ci.keep_library_ == false) && (did_compile))
    {
      DynamicLoader::scirun_loader().cleanup_compile(cih);
    }

    status = true;
  }

  reporter->report_progress( ProgressReporter::Done );

  return status;
}

} // Namespace SCIRun

#endif
