
#include <Packages/Uintah/Core/Exceptions/MaxIteration.h>
#include <Core/Geometry/IntVector.h>
#include <sgi_stl_warnings_off.h>
#include <sstream>
#include <sgi_stl_warnings_on.h>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using namespace std;


MaxIteration::MaxIteration(IntVector c,
                           const int count, 
                           const int n_passes,
                           const int L_indx, 
                           const string mes)
{
  ostringstream s;
  s << " cell"<< c << ", Level " << L_indx
    << ", iter " << count << ", n_passes " << n_passes;

  d_msg =  mes + s.str();
  
#ifdef EXCEPTIONS_CRASH
  cout << d_msg << "\n";
#endif
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






