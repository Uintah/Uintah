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
 *  MetropolisReader.cc: Read a persistent metropolis results from a file
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
#include <Dataflow/Modules/DataIO/GenericReader.h>

namespace SCIRun {

using namespace MIT;
template class GenericReader<ResultsHandle>;

}

namespace MIT {


class MetropolisReader : public GenericReader<ResultsHandle> {
public:
  MetropolisReader(const string& id);
};

extern "C" Module* make_MetropolisReader(const string& id) {
  return new MetropolisReader(id);
}

MetropolisReader::MetropolisReader(const string& id)
  : GenericReader<ResultsHandle>("MetropolisReader", id, "DataIO", "MIT")
{
}

} // namespace MIT
