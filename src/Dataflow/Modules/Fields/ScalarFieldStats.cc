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
 *  ScalarFieldStats: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ScalarFieldStats.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>

namespace SCIRun {


DECLARE_MAKER(ScalarFieldStats)
ScalarFieldStats::ScalarFieldStats(GuiContext* ctx)
  : Module("ScalarFieldStats", ctx, Filter, "FieldsOther", "SCIRun"),
    min_(ctx->subVar("min")), max_(ctx->subVar("max")),
    mean_(ctx->subVar("mean")),
    median_(ctx->subVar("median")),
    sigma_(ctx->subVar("sigma")),
    is_fixed_(ctx->subVar("is_fixed")),
    nbuckets_(ctx->subVar("nbuckets"))
{

}



ScalarFieldStats::~ScalarFieldStats()
{
}

void
ScalarFieldStats::fill_histogram( vector<int>& hits)
{
  ostringstream ostr;
  int nmin, nmax;
  vector<int>::iterator it = hits.begin();
  nmin = nmax = *it;
  ostr << *it;  ++it;
  for(; it != hits.end(); ++it){
    ostr <<" "<<*it;
    nmin = ((nmin < *it) ? nmin : *it );
    nmax = ((nmax > *it) ? nmax : *it );
  }
  ostr <<std::ends;
  string smin( to_string(nmin) );
  string smax( to_string(nmax) );

  string data = ostr.str();
  gui->execute(id + " graph_data " + smin + " "
	       + smax + " " + data );
}

void
ScalarFieldStats::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    return;
  }
  if (!ifieldhandle->query_scalar_interface(this).get_rep())
  {
    error("This module only works on scalar fields.");
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  CompileInfoHandle ci = ScalarFieldStatsAlgo::get_compile_info(ftd, ltd);
  Handle<ScalarFieldStatsAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  algo->execute(ifieldhandle, this);
}



CompileInfoHandle
ScalarFieldStatsAlgo::get_compile_info(const TypeDescription *field_td,
				     const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScalarFieldStatsAlgoT");
  static const string base_class_name("ScalarFieldStatsAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name() + ", " + loc_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
