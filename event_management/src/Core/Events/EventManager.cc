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
//    File   : EventManager.cc
//    Author : Martin Cole
//    Date   : Thu May 25 15:53:19 2006

#include <Core/Events/EventManager.h>
#include <iostream>

namespace SCIRun {

EventManager::id_tm_map_t EventManager::mboxes_;
Mailbox<event_handle_t> EventManager::mailbox_("EventManager", 32);

EventManager::EventManager() :
  tm_("EventManager tools")
{
  cerr << "EventManager has been created." << endl;
}

EventManager::~EventManager()
{
  cerr << "EventManager has been destroyed." << endl;
}

EventManager::event_mailbox_t*
EventManager::register_event_messages(string id)
{
  id_tm_map_t::iterator it = mboxes_.find(id);
  if (it != mboxes_.end()) {
    cerr << "Warning: id: " << id 
	 << " is already registered, and getting events." << endl;
    return mboxes_[id];
  }
  string mbox_name = "event_mailbox_" + id;
  event_mailbox_t* mbox = new event_mailbox_t(mbox_name.c_str(), 32);
  mboxes_[id] = mbox;
  return mbox;
}

void
EventManager::unregister_event_messages(string id)
{
  id_tm_map_t::iterator it = mboxes_.find(id);
  if (it != mboxes_.end()) {
    cerr << "Warning: id: " << id 
	 << " is not registered." << endl;
    return;
  }
  event_mailbox_t* mbox = mboxes_[id];
  delete mbox;
  mboxes_.erase(it);
}

void
EventManager::run() 
{
  bool done = false;
  event_handle_t event;
  do {
    event = tm_.propagate_event(mailbox_.receive());
    if (dynamic_cast<QuitEvent*>(event.get_rep()) != 0) {
      done = true;
    }
    // If the event has a specific target mailbox,
    if (!event->get_target().empty()) {
      mboxes_[event->get_target()]->send(event);
    } else {
      // If the event has no target mailbox, broadcast it to all mailboxes
      id_tm_map_t::iterator it = mboxes_.begin();
      for (;it != mboxes_.end(); ++it) {
        it->second->send(event);
      }
    }
    // If the event is a QuitEvent type, then shutdown the Event Manager
  } while (! done);
}

} // namespace SCIRun
