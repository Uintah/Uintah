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

// Attrib.h - the base attribute class.
//
//  Written by:
//   Eric Kuehne, Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute
//
//  General storage class for Fields.
//

#ifndef SCI_project_NeumannBC_h
#define SCI_project_NeumannBC_h 1

#include <vector>
#include <string>
#include <iostream>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Vector.h>

namespace BioPSE {

using namespace SCIRun;

/////////
// Structure to hold Neumann BC related values
class NeumannBC {
public:  
  // GROUP: public data
  //////////
  // 
  NeumannBC(){};
  NeumannBC(Vector v, double d): dir(v), val(d){};
  //////////
  // Direction to take derivative in
  Vector dir;
  //////////
  // Value of the derivative
  double val;
};
} // end namespace BioPSE


namespace SCIRun {
using namespace std;
//////////
// PIO for NeumannBC objects
void  Pio(Piostream&, BioPSE::NeumannBC&);
ostream& operator<<(ostream&, BioPSE::NeumannBC&);
} // end namespace SCIRun

#endif
