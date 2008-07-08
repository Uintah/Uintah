
#include <Packages/Uintah/StandAlone/tools/puda/util.h>
#include <cstdlib>
#include <iostream>

using namespace std;

void
Uintah::findTimestep_loopLimits( const bool tslow_set, 
                                 const bool tsup_set,
                                 const vector<double> times,
                                 unsigned long & time_step_lower,
                                 unsigned long & time_step_upper )
{
  if( !tslow_set ) {
    time_step_lower = 0;
  }
  else if( time_step_lower >= times.size() ) {
    cerr << "timesteplow must be between 0 and " << times.size()-1 << "\n";
    abort();
  }
  if( !tsup_set ) {
    time_step_upper = times.size() - 1;
  }
  else if( time_step_upper >= times.size() ) {
    cerr << "timestephigh must be between 0 and " << times.size()-1 << "\n";
    abort();
  }
}
