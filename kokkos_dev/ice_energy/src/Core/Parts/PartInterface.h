/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
