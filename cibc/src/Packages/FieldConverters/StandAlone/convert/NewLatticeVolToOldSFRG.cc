/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  OldSFRGtoNewLatVolField.cc: Converter
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
//#include <Core/Datatypes/LatVolField.h>

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
    cerr << "Usage: "<<argv[0]<<" OldSFRG NewLatVolField\n";
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
