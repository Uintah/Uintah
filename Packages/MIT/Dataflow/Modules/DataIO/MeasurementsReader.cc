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

namespace MIT {

using namespace SCIRun;

//template class GenericReader<MeasurementsHandle>;

class MeasurementsReader : public GenericReader<MeasurementsHandle> {
public:
  MeasurementsReader(const string& id);
};

extern "C" Module* make_MeasurementsReader(const string& id) {
  return new MeasurementsReader(id);
}

MeasurementsReader::MeasurementsReader(const string& id)
  : GenericReader<MeasurementsHandle>("MeasurementsReader", id, "DataIO", "MIT")
{
}


} // End namespace MIT


