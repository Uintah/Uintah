

#include <Packages/Uintah/Core/Tracker/TrackerClient.h>

#include <iostream>
#include <sstream>

using namespace std;
using namespace Uintah;

string
Tracker::toString( MPIMessageType mt )
{
  switch( mt ) {
  case MPI_BCAST :
    return "MPI_BCAST";
    break;
  case MPI_SEND :
    return "MPI_SEND";
    break;
  case MPI_RECV :
    return "MPI_RECV";
    break;
  } // end switch( mt )
  return "unknown MPIMessageType :(";
}

string
Tracker::toString( GeneralMessageType mt )
{
  switch( mt ) {
  case VARIABLE_NAME :
    return "VARIABLE_NAME";
    break;
  case TIMESTEP_STARTED :
    return "TIMESTEP_STARTED";
    break;
  } // end switch( mt )
  return "unknown GeneralMessageType... :(";
}
