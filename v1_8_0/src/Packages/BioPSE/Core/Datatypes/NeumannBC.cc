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

// NeumannBC.cc - Neuamann boundary condition
//
//  Written by:
//   Alexei Samsonov
//   Department of Computer Science
//   University of Utah
//   April 2000, December 2000
//
//  Copyright (C) 2000 SCI Institute

#include <Packages/BioPSE/Core/Datatypes/NeumannBC.h>

namespace SCIRun {

//////////
// PIO for NeumannBC objects
void Pio(Piostream& stream, BioPSE::NeumannBC& nmn){
  stream.begin_cheap_delim();
  Pio(stream, nmn.dir);
  Pio(stream, nmn.val);
  stream.end_cheap_delim();
}
//////////
//
ostream& operator<<(ostream& ostr, BioPSE::NeumannBC& nmn){
  ostr << "["<< nmn.dir << ", " << nmn.val << "]";
  return ostr;
}
}  // end namespace SCIRun
