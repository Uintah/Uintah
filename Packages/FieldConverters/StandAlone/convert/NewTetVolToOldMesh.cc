/*
 *  OldSFRGtoNewLatticeVol.cc: Converter
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

//#include <Packages/FieldConverters/Datatypes/ScalarFieldRG.h>
//#include <Core/Datatypes/LatticeVol.h>

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

using std::cerr;
using std::ifstream;
using std::endl;

//using namespace SCIRun;
//using namespace FieldConverters;

int
main(int argc, char **argv) {
#if 0
  ScalarFieldHandle handle;
  
  if (argc !=3) {
    cerr << "Usage: "<<argv[0]<<" OldSFRG NewLatticeVol\n";
    exit(0);
  }
  Piostream* stream=auto_istream(argv[1]);
  if (!stream) {
    cerr << "Error - couldn't open file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  Pio(*stream, handle);
  if (!handle.get_rep()) {
    cerr << "Error reading ScalarField from file "<<argv[1]<<".  Exiting...\n";
    exit(0);
  }
  
  ScalarFieldRGBase *base=dynamic_cast<ScalarFieldRGBase*>(handle.get_rep());
  if (!base) {
    cerr << "Error - input Field wasn't an SFRG.\n";
    exit(0);
  }

  FieldHandle fH;

  // TO_DO:
  // make a new Field, set it to fH, and give it a mesh and data like base's

  BinaryPiostream stream(argv[2], Piostream::Write);
  Pio(stream, fH);
#endif
  return 0;  
}    
