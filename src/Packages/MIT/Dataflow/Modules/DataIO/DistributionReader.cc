/*
 *  DistributionReader.cc:
 *
 *  Written by:
 *   Yarden Livnat
 *   July 2001
 *
 */

#include <Packages/MIT/Dataflow/Ports/MetropolisPorts.h>
#include <Dataflow/Modules/DataIO/GenericReader.h>

namespace SCIRun {

using namespace MIT;
template class GenericReader<DistributionHandle>;

}


namespace MIT {

using namespace SCIRun;


class DistributionReader : public GenericReader<DistributionHandle> {
public:
  DistributionReader(const string& id);
};

extern "C" Module* make_DistributionReader(const string& id) {
  return new DistributionReader(id);
}

DistributionReader::DistributionReader(const string& id)
  : GenericReader<DistributionHandle>("DistributionReader", id, "DataIO", "MIT")
{
}


} // End namespace MIT


