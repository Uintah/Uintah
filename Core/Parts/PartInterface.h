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
 *  Interface.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Deinterfacement of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_PartInterface_h
#define SCI_PartInterface_h 

#include <vector>
#include <Core/Util/Signals.h>

namespace SCIRun {
  
class Part;
class SciEvent;

class PartInterface {
protected:
  Part *part_;
  PartInterface *parent_;
  vector<PartInterface *> children_;

public:
  PartInterface( Part *, PartInterface *parent );
  virtual ~PartInterface();
  
  Part *part() { return part_; }
  
  void add_child( PartInterface *);
  void rem_child( PartInterface *);
  
  Signal1<PartInterface *> has_child;
  Signal1<SciEvent *> report;
};

} // namespace SCIRun

#endif // SCI_Interface_h
