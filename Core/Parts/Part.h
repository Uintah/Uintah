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
 *  Part.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Part_h
#define SCI_Part_h 

#include <string>

namespace SCIRun {
  
class PartInterface;

class Part {
protected:
  PartInterface *parent_interface_;
  PartInterface *interface_;

  string name_;

public:
  Part( PartInterface *parent=0, const string name="" ) 
    : parent_interface_(parent), name_(name) {}
  virtual ~Part() {/* if (interface_) delete interface_;*/ }

  PartInterface *interface() { return interface_; }
  string name() { return name_; }
};

} // namespace SCIRun

#endif // SCI_Part_h
