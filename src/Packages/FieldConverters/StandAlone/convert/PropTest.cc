/*
 *  PropTest.cc: test stuff
 *
 *  Written by:
 *   Dave Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Datatypes/TetVol.h>
#include <Core/Containers/Array1.h>

using namespace SCIRun;

int
main(int argc, char **argv)
{  
  TetVol<double> *field = new TetVol<double>;
  field->store("date", string("today"));
  Array1<double> temp;
  field->store("junk", temp);
  delete(field);
  return 0;  
}    
