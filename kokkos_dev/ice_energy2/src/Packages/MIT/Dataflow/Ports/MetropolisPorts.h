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

// MeasurmentsPort.h
//  Written by:
//   Yarden Livnat
//   Department of Computer Science
//   University of Utah
//   July 2001
//  Copyright (C) 2001 SCI Group

#ifndef MIT_FieldPort_h
#define MIT_FieldPort_h 

#include <Dataflow/Network/Ports/SimplePort.h>
#include <Packages/MIT/Core/Datatypes/MetropolisData.h>

namespace MIT {
    
using namespace SCIRun;

typedef SimpleIPort<MeasurementsHandle> MeasurementsIPort;
typedef SimpleOPort<MeasurementsHandle> MeasurementsOPort;

typedef SimpleIPort<DistributionHandle> DistributionIPort;
typedef SimpleOPort<DistributionHandle> DistributionOPort;

typedef SimpleIPort<ResultsHandle> ResultsIPort;
typedef SimpleOPort<ResultsHandle> ResultsOPort;

} // End namespace MIT

#endif
