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
 *  HexToTet.cc:  Convert a Hex field into a Tet field using 1-5 split
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Modules/Fields/HexToTet.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DynamicCompilation.h>

#include <iostream>
#include <vector>
#include <algorithm>


namespace SCIRun {

class HexToTet : public Module {
private:
  int last_generation_;
  FieldHandle ofieldhandle;

public:

  //! Constructor/Destructor
  HexToTet(GuiContext *context);
  virtual ~HexToTet();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(HexToTet)


HexToTet::HexToTet(GuiContext *context) : 
  Module("HexToTet", context, Filter, "FieldsGeometry", "SCIRun"),
  last_generation_(0)
{
}

HexToTet::~HexToTet()
{
}

void
HexToTet::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("HexVol");
  FieldHandle ifieldhandle;
  if(!(ifieldport->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Can't get field.");
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("TetVol");

  // Cache generation.
  if (ifieldhandle->generation == last_generation_)
  {
    ofp->send(ofieldhandle);
    return;
  }
  last_generation_ = ifieldhandle->generation;
  const TypeDescription *src_td = ifieldhandle->get_type_description();
  CompileInfoHandle hci = HexToTetAlgo::get_compile_info(src_td);
  Handle<HexToTetAlgo> halgo;
  if (DynamicCompilation::compile(hci, halgo, true, this))
  {
    if (!halgo->execute(ifieldhandle, ofieldhandle, this))
    {
      warning("HexToTet conversion failed to copy data.");
      return;
    }
  }
  else
  {
    CompileInfoHandle lci = LatToTetAlgo::get_compile_info(src_td);
    Handle<LatToTetAlgo> lalgo;
    if (DynamicCompilation::compile(lci, lalgo, true, this))
    {
      if (!lalgo->execute(ifieldhandle, ofieldhandle, this))
      {
	warning("LatToTet conversion failed to copy data.");
	return;
      }
    }
    else
    {
      error("HexToTet only supported for Hex types -- failed for "+
	    src_td->get_name());
      return;
    }
  }
  ofp->send(ofieldhandle);
}


CompileInfoHandle
HexToTetAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("HexToTetAlgoT");
  static const string base_class_name("HexToTetAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       src_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       src_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  src_td->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
LatToTetAlgo::get_compile_info(const TypeDescription *src_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("LatToTetAlgoT");
  static const string base_class_name("LatToTetAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       src_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       src_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  src_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
