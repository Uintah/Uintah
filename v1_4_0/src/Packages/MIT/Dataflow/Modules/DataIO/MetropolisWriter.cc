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
 *  MetropolisWriter.cc: Save persistent representation of metropolis resutls
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>
#include <Dataflow/Modules/DataIO/GenericWriter.h>


namespace SCIRun {

using namespace MIT;
template class GenericWriter<ResultsHandle>;

}

namespace MIT {

using namespace SCIRun;

class MetropolisWriter : public GenericWriter<ResultsHandle> {
public:
  MetropolisWriter(const string& id);
};


extern "C" Module* make_MetropolisWriter(const string& id) {
  return new MetropolisWriter(id);
}


MetropolisWriter::MetropolisWriter(const string& id)
  : GenericWriter<ResultsHandle>("MetropolisWriter", id, "DataIO", "MIT")
{
}

} // namespace MIT
