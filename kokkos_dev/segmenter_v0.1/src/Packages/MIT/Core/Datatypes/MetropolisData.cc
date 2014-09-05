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


#include <Packages/MIT/Core/Datatypes/MetropolisData.h>
#include <Core/Malloc/Allocator.h>

namespace MIT {
  
using namespace SCIRun;

static Persistent* make_Measurements()
{
  return scinew Measurements;
}

// initialize the static member type_id
PersistentTypeID Measurements::type_id("Measurements", "Datatype", 
				       make_Measurements);

const int MEASURMENTS_VERSION = 1;

void
Measurements::io(Piostream& stream){

  stream.begin_class("Measurements", MEASURMENTS_VERSION);
  Pio(stream, t);
  Pio(stream, concentration );
  stream.end_class();
}

static Persistent* make_Distribution()
{
    return scinew Distribution;
}

PersistentTypeID Distribution::type_id("Distribution", "Datatype", 
				       make_Distribution);

const int DISTRIBUTION_VERSION = 1;

void 
Distribution::io(Piostream& stream){

  stream.begin_class("Distribution", DISTRIBUTION_VERSION);
  Pio(stream, sigma);
  Pio(stream, kappa );
  Pio(stream, theta );
  stream.end_class();
}

static Persistent* make_Results()
{
    return scinew Results;
}

// initialize the static member type_id
PersistentTypeID Results::type_id("Results", "Datatype", make_Results);

const int RESULTS_VERSION = 1;

void
Results::io(Piostream& stream){

 stream.begin_class("Results", RESULTS_VERSION);
  Pio(stream, k_);
  Pio(stream, data_ );
  Pio(stream, color_ );
  stream.end_class();
}
}
