
#include <Packages/Uintah/Core/Tracker/TrackerServer.h>

#include <iostream>

using namespace std;
using namespace Uintah;

int
main()
{
  int numClients = 1;

  cout << "Starting to track sus running on " << numClients << " hosts.\n";

  TrackerServer::startTracking( numClients );

  cout << "hit any key to quit\n";

  char answer;
  cin >> answer;

  TrackerServer::quit();
}
