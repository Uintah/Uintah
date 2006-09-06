//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
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
//    File   : TkOpenGLEventSpawner.cc
//    Author : McKay Davis
//    Date   : Fri Jun  2 13:37:16 MDT 2006

#include <Core/Events/BaseEvent.h>
#include <Dataflow/GuiInterface/TCLInterface.h>
#include <Dataflow/GuiInterface/TkOpenGLEventSpawner.h>
#include <Dataflow/GuiInterface/TkOpenGLContext.h>

namespace SCIRun {
  TkOpenGLEventSpawner::TkOpenGLEventSpawner(TkOpenGLContext *context) :
    EventSpawner("Tk"),
    context_(context)
  {
    ASSERT(context);
    string path = "{"+context_->id_+"}";
    string command = context_->id_+"-c";
    
    GuiInterface * gui = GuiInterface::getSingleton();
    gui->add_command(command, this, 0);
    command = "{"+command+"}";
    gui->eval("bind "+path+" <Motion> \""+
              command+" Motion %W %b %s %X %Y %x %y %t\"");
    gui->eval("bind "+path+" <ButtonPress> \""+
              command+" ButtonPress %W %b %s %X %Y %x %y %t\"");
    gui->eval("bind "+path+" <ButtonRelease> \""+
              command+" ButtonRelease %W %b %s %X %Y %x %y %t\"");
    gui->eval("bind "+path+" <KeyPress> \""+
              command+" KeyPress %W %b %s %X %Y %x %y %t\"");
    gui->eval("bind "+path+" <KeyRelease> \""+
              command+" KeyRelease %W %b %s %X %Y %x %y %t\"");
    gui->eval("bind "+path+" <Expose> \""+
              command+" Expose %W\"");
    gui->eval("bind "+path+" <Configure> \""+
              command+" Configure %W\"");
    gui->eval("bind "+path+" <Destroy> \""+
              command+" Destroy %W\"");
    gui->eval("bind "+path+" <Enter> \"focus %W;"+
              command+" Enter %W %b %s %X %Y %x %y %t\"");
    gui->eval("bind "+path+" <Leave> \""+
              command+" Leave %W %b %s %X %Y %x %y %t\"");
    
  }
  
  TkOpenGLEventSpawner::~TkOpenGLEventSpawner() {
  }
  
  void
  TkOpenGLEventSpawner::tcl_command(GuiArgs &args, void *) 
  {
#if 0
    if (sci_getenv_p("TCL_DEBUG")) {
      cerr << "TCL> ";
      for (int a = 0; a < args.count(); ++a)
        cerr << args[a] << " ";
      cerr << "\n";
    }
    
    EventState *eventptr = new EventState;
    EventState &event = *eventptr;
    
    if (args[1] == "Motion")
      event.type_ = EventState::MOTION_E;
    else if (args[1] == "ButtonPress")
      event.type_ = EventState::BUTTON_PRESS_E;
    else if (args[1] == "ButtonRelease")
      event.type_ = EventState::BUTTON_RELEASE_E;
    else if (args[1] == "KeyPress")
      event.type_ = EventState::KEY_PRESS_E;
    else if (args[1] == "KeyRelease")
      event.type_ = EventState::KEY_RELEASE_E;
    else if (args[1] == "Expose")
      event.type_ = EventState::EXPOSE_E;
    else if (args[1] == "Configure")
      event.type_ = EventState::CONFIGURE_E;
    else if (args[1] == "Destroy")
      event.type_ = EventState::DESTROY_E;
    else if (args[1] == "Enter")
      event.type_ = EventState::ENTER_E;
    else if (args[1] == "Leave")
      event.type_ = EventState::LEAVE_E;
    
    if (event.type_ == EventState::EXPOSE_E || 
        event.type_ == EventState::CONFIGURE_E || 
        event.type_ == EventState::DESTROY_E) {
      return;
    }
    
    event.state_ = args.get_int(4);
    
    if (event.type_ == EventState::KEY_PRESS_E || 
        event.type_ == EventState::KEY_RELEASE_E) {
      TCLKeysym_t::iterator keysym = TCLKeysym::tcl_keysym.find(args[3]);
      
      if (keysym != TCLKeysym::tcl_keysym.end()) {
        event.key_ = "";
        event.key_.push_back(keysym->second);
      } else {
        event.key_ = args[4];
      }
      
      
      if (event.type_ == EventState::KEY_RELEASE_E)
        event.keys_.insert(event.key_);
      else {
        set<string>::iterator pos = event.keys_.find(event.key_);
        if (pos != event.keys_.end())
          event.keys_.erase(pos);
        else {
          return;
        }
      }
      return;
    }
    
    if (event.type_ != EventState::MOTION_E || 
        event.type_ != EventState::ENTER_E || 
        event.type_ != EventState::LEAVE_E) {
      // The button parameter may be invalid on motion events (mainly OS X)
      // Button presses don't set state correctly, so manually set state_ here
      // to make Event::button() method work on press events
      event.button_ = args.get_int(3);
      switch (event.button_) {
      case 1: event.state_ |= EventState::BUTTON_1_E; break;
      case 2: event.state_ |= EventState::BUTTON_2_E; break;
      case 3: event.state_ |= EventState::BUTTON_3_E; break;
      case 4: event.state_ |= EventState::BUTTON_4_E; break;
      case 5: event.state_ |= EventState::BUTTON_5_E; break;
      default: break;
      }
    }
    
    event.X_ = args.get_int(5);
    event.Y_ = args.get_int(6);
    event.x_ = args.get_int(7);
    event.y_ = context_->height() - 1 - args.get_int(8);
    
    event_manager_->add_event(eventptr);
#endif
  }
}
