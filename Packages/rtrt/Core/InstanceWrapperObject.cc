
#include <Packages/rtrt/Core/InstanceWrapperObject.h>
#include <values.h>

using namespace rtrt;
using namespace SCIRun;

const int INSTANCEWRAPPEROBJECT_VERSION = 1;
void 
InstanceWrapperObject::io(SCIRun::Piostream &str)
{
  str.begin_class("InstanceWrapperObject", INSTANCEWRAPPEROBJECT_VERSION);
  str.end_class();
}

namespace SCIRun {
void 
Pio(SCIRun::Piostream &str, rtrt::InstanceWrapperObject *&i)
{
  str.begin_cheap_delim();
  str.end_cheap_delim();
}
} // end namespace SCIRun
