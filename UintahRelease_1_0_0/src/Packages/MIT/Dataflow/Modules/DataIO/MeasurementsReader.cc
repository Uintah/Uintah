/*
 *  MeasurementsReader.cc:
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
template class GenericReader<MeasurementsHandle>;
  
}

namespace MIT {

using namespace SCIRun;

class MeasurementsReader : public GenericReader<MeasurementsHandle> {
public:
  MeasurementsReader(const string& get_id());
};

extern "C" Module* make_MeasurementsReader(const string& get_id()) {
  return new MeasurementsReader(get_id());
}

MeasurementsReader::MeasurementsReader(const string& get_id())
  : GenericReader<MeasurementsHandle>("MeasurementsReader", get_id(), "DataIO", "MIT")
{
}


} // End namespace MIT


