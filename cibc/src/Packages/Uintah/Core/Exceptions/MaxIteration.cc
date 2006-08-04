
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
                           const string message,
                           const char* file,
                           int line)
{
  ostringstream s;
  s << "A MaxIteration exception was thrown.\n"
    << file << ":" << line << "\n" << message
    << " cell"<< c << ", Level " << L_indx
    << ", iter " << count << ", Timestep " << n_passes 
    << "\n\n This usually means that something much deeper has gone wrong with the simulation."
    << "\n Compute equilibration pressure task is rarely the problem";

  d_msg = s.str();
  
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






