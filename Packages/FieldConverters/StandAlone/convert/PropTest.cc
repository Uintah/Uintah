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

using namespace SCIRun;

main(int argc, char **argv) {  
  TetVol<double> *field = new TetVol<double>;
  field->set_string("date", "today");
  delete(field);
  return 0;  
}    
