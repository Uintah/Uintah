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
#include <Core/Parts/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>
#include <strstream>
#include <string>
#include <stdio.h>

namespace SCIRun {


extern "C" Module* make_ScalarFieldStats(const string& id)
{
  return new ScalarFieldStats(id);
}

ScalarFieldStats::ScalarFieldStats(const string& id)
  : Module("ScalarFieldStats", id, Filter, "Fields", "SCIRun"),
    min_("min", id, this), max_("max", id, this),
    mean_("mean", id, this),
    median_("median", id, this),
    sigma_("sigma", id, this),
    is_fixed_("is_fixed", id, this),
    nbuckets_("nbuckets", id, this)
{

}



ScalarFieldStats::~ScalarFieldStats()
{
}

void
ScalarFieldStats::fill_histogram( vector<int>& hits)
{
  ostrstream ostr;
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

  char *data = ostr.str();
  tcl_execute(id + " graph_data " + smin.c_str() + " "
	       + smax.c_str() + " " + data );
  
  delete data;
}

void
ScalarFieldStats::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()
	&& ifieldhandle->is_scalar()))
  {
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->data_at_type_description();
  CompileInfo *ci = ScalarFieldStatsAlgo::get_compile_info(ftd, ltd);
  DynamicAlgoHandle algo_handle;
  if (! DynamicLoader::scirun_loader().get(*ci, algo_handle))
  {
    cout << "Could not compile algorithm." << std::endl;
    return;
  }
  ScalarFieldStatsAlgo *algo =
    dynamic_cast<ScalarFieldStatsAlgo *>(algo_handle.get_rep());
  if (algo == 0)
  {
    cout << "Could not get algorithm." << std::endl;
    return;
  }
  algo->execute(ifieldhandle, this);

}



CompileInfo *
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
