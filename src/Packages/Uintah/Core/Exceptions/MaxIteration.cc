
#include <Packages/Uintah/Core/Exceptions/MaxIteration.h>
#include <Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


MaxIteration::MaxIteration(IntVector c,int count, int n_passes, string mes)
{
  ostringstream s;
  s << " cell["<<c.x()<<"]["<<c.y()<<"]["<<c.z()
    << "], iter " << count << ", n_passes " << n_passes;

  d_msg =  mes + s.str();
  
}


MaxIteration::MaxIteration(const MaxIteration& copy)
  : d_msg(copy.d_msg)
{
}

MaxIteration::~MaxIteration()
{
}

const char* MaxIteration::message() const
{
  return d_msg.c_str();
}

const char* MaxIteration::type() const
{
  return "Packages/Uintah::Exceptions::MaxIteration";
}






