
#include <Packages/rtrt/Core/Instance.h>
#include <values.h>

using namespace rtrt;
using namespace SCIRun;

const int INSTANCE_VERSION = 1;
void 
Instance::io(SCIRun::Piostream &str)
{
  str.begin_class("Instance", INSTANCE_VERSION);
  Object::io(str);
  str.end_class();
}

namespace SCIRun {
void 
Pio(SCIRun::Piostream &str, rtrt::Instance *&i)
{
  str.begin_cheap_delim();
  str.end_cheap_delim();
}
} // end namespace SCIRun
