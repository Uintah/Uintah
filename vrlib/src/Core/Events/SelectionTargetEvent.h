//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : SelectionTargetEvent.h
//    Author : Martin Cole
//    Date   : Fri Jul 28 11:37:11 2006


#if !defined(SelectionTargetEvent_h)
#define SelectionTargetEvent_h

#include <Core/Events/BaseEvent.h>
#include <Core/Datatypes/Field.h>
#include <string>


namespace SCIRun {

using namespace std;

class SelectionTargetEvent : public BaseEvent
{
public:
  //! target is the id of the mailbox given by EventManager, 
  //! if it is an empty string then all Event Mailboxes get the event.
  SelectionTargetEvent(const string &target = "", 
		       long int time = 0);


  virtual ~SelectionTargetEvent();

  virtual SelectionTargetEvent *clone() { 
    return new SelectionTargetEvent(*this); 
  }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  
  FieldHandle   get_selection_target() const         { return sel_targ_; }
  int           get_selection_id()     const         { return sel_id_; }

  void          set_selection_target(FieldHandle fh) { sel_targ_ = fh; }
  void          set_selection_id(int id)             { sel_id_ = id; }

private:
  //! The event timestamp
  FieldHandle           sel_targ_;
  int                   sel_id_;
};

} // namespace SCIRun

#endif // SelectionTargetEvent_h
