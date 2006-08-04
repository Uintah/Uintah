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

// MetropolisPorts.cc
//  Written by:
//   Yarden Livnat
//   Department of Computer Science
//   University of Utah
//   July 2001
//  Copyright (C) 2001 SCI Group


#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>
#include <Core/Malloc/Allocator.h>

namespace MIT {

using namespace SCIRun;

extern "C" {
PSECORESHARE IPort* make_MeasurementsIPort(Module* module, const string& name) {
  return scinew SimpleIPort<MeasurementsHandle>(module,name);
}
PSECORESHARE OPort* make_MeasurementsOPort(Module* module, const string& name) {
  return scinew SimpleOPort<MeasurementsHandle>(module,name);
}


PSECORESHARE IPort* make_DistributionIPort(Module* module, const string& name) {
  return scinew SimpleIPort<DistributionHandle>(module,name);
}
PSECORESHARE OPort* make_DistributionOPort(Module* module, const string& name) {
  return scinew SimpleOPort<DistributionHandle>(module,name);
}

PSECORESHARE IPort* make_ResultsIPort(Module* module, const string& name) {
  return scinew SimpleIPort<ResultsHandle>(module,name);
}
PSECORESHARE OPort* make_ResultsOPort(Module* module, const string& name) {
  return scinew SimpleOPort<ResultsHandle>(module,name);
}
}

} // End namespace MIT

namespace SCIRun {

  using namespace MIT;
  
template<> string SimpleIPort<MeasurementsHandle>::port_type_("Measurements");
template<> string SimpleIPort<MeasurementsHandle>::port_color_("yellow");

template<> string SimpleIPort<DistributionHandle>::port_type_("Distribution");
template<> string SimpleIPort<DistributionHandle>::port_color_("green");

template<> string SimpleIPort<ResultsHandle>::port_type_("Results");
template<> string SimpleIPort<ResultsHandle>::port_color_("white");

} // End namespace SCIRun
