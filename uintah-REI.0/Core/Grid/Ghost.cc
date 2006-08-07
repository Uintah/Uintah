#include <Packages/Uintah/Core/Grid/Ghost.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

string Ghost::names[numGhostTypes] =
{ "None", "nodes", "cells", "x faces", "y faces", "z faces", "all faces" };
  

IntVector Ghost::directions[numGhostTypes] =
{ IntVector(0,0,0), IntVector(1,1,1), IntVector(1,1,1),
  IntVector(1,0,0), IntVector(0,1,0), IntVector(0,0,1),
  IntVector(1,1,1) };
