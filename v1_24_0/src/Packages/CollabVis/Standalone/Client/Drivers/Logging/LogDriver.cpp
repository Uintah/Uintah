#include <Logging/Log.h>
#include <iostream>

using namespace SemotusVisum;

void
doLog() {
  Log::log( DEBUG  , "Debug info" );
  Log::log( MESSAGE, "Message info" );
  Log::log( WARNING, "Warning info" );
  Log::log( ERROR,   "Error info" );
  
}


int
main() {

  std::cerr << "Opening logfile named Log.1" << endl;
  Log::setLogFileName( "Log.1" );
  doLog();
  Log::close();
  
  std::cerr << "Opening default logfile:" << endl;
  doLog();

  Log::setLogLevel( MESSAGE );
  Log::log( MESSAGE, "You should only see message and above now");
  doLog();
  
  Log::setLogLevel( WARNING );
  Log::log( WARNING, "You should only see warning and above now");
  doLog();

  Log::setLogLevel( ERROR );
  Log::log( ERROR, "You should only see error and above now");
  doLog();

  Log::setLogLevel( DEBUG );
  Log::log( DEBUG, "You should only see debug and above now");
  doLog();
}
