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
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_PartInterface_h
#define SCI_PartInterface_h 

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Signals.h>
#include <Core/Parts/Part.h>

namespace SCIRun {
using std::string;
  
class SciEvent;

class PartInterface {
protected:
  string type_;
  Part *part_;
  PartInterface *parent_;
  vector<PartInterface *> children_;

public:
  PartInterface( Part *, PartInterface *parent, const string &type, bool=true);
  virtual ~PartInterface();

  virtual void init();
  Part *part() { return part_; }
  
  string name();

  void set_parent( PartInterface * );
  void add_child( PartInterface *);
  void rem_child( PartInterface *);
  string type() { return type_; }

  template<class T, class Arg>
  void report_children( T &t, void (T::*fun)(Arg) );

  template<class T, class Arg>
  void report_children( T *t, void (T::*fun)(Arg) );

  virtual void report_children( SlotBase1<PartInterface *> &slot );

  void set_property(int id, const string &name, vector<unsigned char> data) 
  { 
    if (part_)
      part_->set_property(id,name,data);
  }
  void get_property(int id, const string &name , vector<unsigned char> &data) 
  { 
    if (part_)
      part_->get_property(id,name,data);
  }

  // Slots
  Signal1<PartInterface *> has_child;
  Signal1<SciEvent *> report;
};


template<class T, class Arg>
void PartInterface::report_children( T *t, void (T::*fun)(Arg) )
{
  Slot1<T,Arg> slot(t, fun);
  report_children( slot );
}


template<class T, class Arg>
void PartInterface::report_children( T &t, void (T::*fun)(Arg) )
{
  Slot1<T,Arg> slot(&t, fun);
  report_children( slot );
}

} // namespace SCIRun

#endif // SCI_Interface_h
