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
  DistributionReader(const string& get_id());
};

extern "C" Module* make_DistributionReader(const string& get_id()) {
  return new DistributionReader(get_id());
}

DistributionReader::DistributionReader(const string& get_id())
  : GenericReader<DistributionHandle>("DistributionReader", get_id(), "DataIO", "MIT")
{
}


} // End namespace MIT


