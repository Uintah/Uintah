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
 *  ModuleReporter.h: Interface for updating a module's status.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2002
 *
 *  Copyright (C) 2002 SCI Group
 */


#ifndef SCIRun_Core_Util_ModuleReporter_h
#define SCIRun_Core_Util_ModuleReporter_h

#include <Core/Util/DynamicLoader.h>
#include <iostream>

namespace SCIRun {

class SCICORESHARE ModuleReporter
{
public:
  virtual ~ModuleReporter();

  virtual void error(const std::string&) = 0;
  virtual void warning(const std::string&) = 0;
  virtual void remark(const std::string&) = 0;
  virtual void postMessage(const std::string&) = 0;

  virtual std::ostream &msgStream() = 0;
  virtual void msgStream_flush() = 0;

  template <class DC>
  bool module_dynamic_compile(CompileInfoHandle ci, DC &result);

  template <class DC>
  bool module_maybe_dynamic_compile(CompileInfoHandle ci, DC &result);

protected:
  virtual void light_module0() = 0;
  virtual void light_module() = 0;
  virtual void reset_module_color() = 0;
};


template <class DC>
bool
ModuleReporter::module_dynamic_compile(CompileInfoHandle cih, DC &result)
{
  ASSERT(cih.get_rep());
  const CompileInfo &ci = *(cih.get_rep());
  DynamicAlgoHandle algo_handle;
  light_module0();
  bool reset_color = false;
  if (! DynamicLoader::scirun_loader().fetch(ci, algo_handle))
  {
    remark("Dynamically compiling some code.");
    light_module();
    const bool status =
      DynamicLoader::scirun_loader().compile_and_store(ci, false, msgStream());
    reset_module_color();
    reset_color = true;
    msgStream_flush();
    remark("Dynamic compilation completed.");

    if (! (status && DynamicLoader::scirun_loader().fetch(ci, algo_handle)))
    {
      error("Could not compile algorithm for '" +
	    ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
      return false;
    }
  }

  if (!reset_color) { reset_module_color(); }

  result = dynamic_cast<typename DC::pointer_type>(algo_handle.get_rep());
  if (result.get_rep() == 0) 
  {
    error("Could not get algorithm for '" +
	  ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
    return false;
  }
  return true;
}


template <class DC>
bool
ModuleReporter::module_maybe_dynamic_compile(CompileInfoHandle cih, DC &result)
{
  ASSERT(cih.get_rep());
  const CompileInfo &ci = *(cih.get_rep());
  DynamicAlgoHandle algo_handle;
  light_module0();
  bool reset_color = false;
  if (! DynamicLoader::scirun_loader().fetch(ci, algo_handle))
  {
    remark("Maybe dynamically compiling some code, ignore failure here.");
    light_module();
    const bool status =
      DynamicLoader::scirun_loader().compile_and_store(ci, true, msgStream());
    reset_module_color();
    reset_color = true;
    msgStream_flush();
    remark("Dynamic compilation completed.");

    if (! (status && DynamicLoader::scirun_loader().fetch(ci, algo_handle)))
    {
      error("Could not compile algorithm for '" +
    	    ci.template_class_name_ + "<" + ci.template_arg_ + ">'.");
      return false;
    }
  }

  if (!reset_color) { reset_module_color(); }

  result = dynamic_cast<typename DC::pointer_type>(algo_handle.get_rep());
  if (result.get_rep() == 0) 
  {
    return false;
  }
  return true;
}

} // Namespace SCIRun

#endif
