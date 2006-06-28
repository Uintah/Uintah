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
#include <Core/Util/Environment.h>
#include <iostream>

namespace SCIRun {

EventManager::id_tm_map_t EventManager::mboxes_;
Mutex EventManager::mboxes_lock_("EventManager mboxes_ lock");
Mailbox<event_handle_t> EventManager::mailbox_("EventManager", 1024);
Piostream * EventManager::stream_ = 0;

EventManager::EventManager() :
  tm_("EventManager tools")
{
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << "EventManager has been created." << endl;
  }
}

EventManager::~EventManager()
{
  mboxes_lock_.lock();
  bool empty = mboxes_.empty();
  mboxes_lock_.unlock();
  if (!empty) {
    TimeThrottle timer;
    timer.start();
    const double start = timer.time();
    double t = start;
    do { 
      mboxes_lock_.lock();
      empty = mboxes_.empty();
      mboxes_lock_.unlock();    
      t = timer.time();
      if (!empty) {
        timer.wait_for_time(t + 1/20.0);
      }
    } while (!empty && (t < start + 5.0));
    timer.stop();
    
    if (!empty && sci_getenv_p("SCI_DEBUG")) {
      cerr << "EventManager quit after waiting " << t - start
           << " seconds for the following mailboxes to unregister:\n";
      
      mboxes_lock_.lock();
      for (id_tm_map_t::iterator it = mboxes_.begin(); 
           it != mboxes_.end(); ++it)
        {
          cerr << it->first << "\n";
        }
      mboxes_lock_.unlock();
    }
  }

  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << "EventManager has been destroyed." << endl;
  }
}


void
EventManager::add_event(event_handle_t event) 
{
  if (stream_ && stream_->reading()) return;

  mailbox_.send(event);
}

bool
EventManager::record_trail_file(const string &filename)
{
  if (stream_) {
    return false;
  }

  stream_ = auto_ostream(filename, "Text", 0);

  if (stream_ && stream_->error()) {
    delete stream_;
    return false;
  }
  stream_->disable_pointer_hashing();

  return stream_;
}

bool
EventManager::play_trail_file(const string &filename)
{
  if (stream_) {
    return false;
  }

  stream_ = auto_istream(filename, 0);
  
  if (stream_ && stream_->error()) {
    delete stream_;
    return false;
  }

  return stream_;

}


void
EventManager::stop_trail_file()
{
  if (!stream_) return;
  delete stream_;
  stream_ = 0;
}
  

EventManager::event_mailbox_t*
EventManager::register_event_messages(string id)
{
  mboxes_lock_.lock();
  id_tm_map_t::iterator it = mboxes_.find(id);
//   if (it != mboxes_.end()) {
//     cerr << "Warning: id: " << id 
// 	 << " is already registered, and getting events." << endl;
//     //    return mboxes_[id];
//   }
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << "Mailbox id \"" << id << "\" registered\n";
  }
  string mbox_name = "event_mailbox_" + id;
  event_mailbox_t* mbox = new event_mailbox_t(mbox_name.c_str(), 1024);
  mboxes_.insert(make_pair(id, mbox));
  mboxes_lock_.unlock();
  return mbox;
}

void
EventManager::unregister_event_messages(string id)
{
  mboxes_lock_.lock();
  id_tm_map_t::iterator it = mboxes_.find(id);
  if (it == mboxes_.end()) {
    cerr << "Warning: maiboxl id """ << id 
         << """ is not registered." << endl;
    cerr << "Valid ids are: ";
    
    for (id_tm_map_t::iterator it = mboxes_.begin(); 
         it != mboxes_.end(); ++it)
      cerr << it->first << ", ";
    cerr << std::endl;
    mboxes_lock_.unlock();
    return;
  }
  if (sci_getenv_p("SCI_DEBUG")) {
    cerr << "Mailbox id \"" << id << "\" un-registered\n";
  }
  delete it->second;
  mboxes_.erase(it);
  mboxes_lock_.unlock();
}


void
EventManager::unregister_mailbox(event_mailbox_t *mailbox)
{
  mboxes_lock_.lock();
  for (id_tm_map_t::iterator it = mboxes_.begin(); it != mboxes_.end(); ++it)
  {
    if (it->second == mailbox) {
      if (sci_getenv_p("SCI_DEBUG")) {
        cerr << "Mailbox id \"" << it->first << "\" un-registered\n";
      }
      delete it->second;
      mboxes_.erase(it);
      mboxes_lock_.unlock();
      return;
    }
  }

  cerr << "Cannot find mailbox to unregister!\n";
  mboxes_lock_.unlock();
}


void
EventManager::play_trail() {
  ASSERT(stream_ && stream_->reading());

  event_handle_t event;
  unsigned long event_time = 0;
  unsigned long last_event_time = 0;
  double last_timer_time = 0;
  const double millisecond = 1.0 / 1000.0;
  TimeThrottle timer;
  timer.start();
  sci_putenv("SCIRUN_TRAIL_PLAYBACK", "1");
  while (stream_ && !stream_->eof()) {
    event = 0;
    Pio(*stream_, event);
  
    if (!event.get_rep()) {
      stop_trail_file();
      continue;
    }   

    if (event_time = event->get_time()) {
      if (last_event_time) {
        double diff = (event_time-last_event_time) * millisecond;
        timer.wait_for_time(last_timer_time + diff);
      }           
      last_event_time = event_time;
      last_timer_time = timer.time();
    }
    
    mailbox_.send(event);
  }
  timer.stop();
  sci_putenv("SCIRUN_TRAIL_PLAYBACK", "0");
} 
  

void
EventManager::run() 
{
  bool done = false;
  event_handle_t event;
  do {
    event = tm_.propagate_event(mailbox_.receive());
    if (stream_ && stream_->writing()) {
      Pio(*stream_, event);
    }

    if (dynamic_cast<QuitEvent*>(event.get_rep()) != 0 &&
        event->get_target().empty()) {
      done = true;
    }

    mboxes_lock_.lock();
    const string target = event->get_target();
    // If the event has a specific target mailbox
    if (!target.empty()) {
      typedef pair<id_tm_map_t::iterator, id_tm_map_t::iterator> mbox_range_t;
      mbox_range_t range = mboxes_.equal_range(target);
      if (range.first == range.second) {
        if (sci_getenv_p("SCI_DEBUG")) {
          cerr << "Event target mailbox id """ << target << """ not found.\n";
        }
      } else {
        for (;range.first != range.second; ++range.first) {
          if (sci_getenv_p("SCI_DEBUG")) {
            cerr << "Event target mailbox id """ << target << """\n";
          }

          if (sci_getenv_p("SCI_DEBUG")) {
            cerr << range.first->first << " size: " 
                 << range.first->second->numItems() << "\n";
          }
          range.first->second->send(event);
        }
      }
    } else {
      if (sci_getenv_p("SCI_DEBUG")) {
        cerr << "Event target mailbox id """"\n";
      }

      id_tm_map_t::iterator it = mboxes_.begin();
      for (;it != mboxes_.end(); ++it) {
        if (sci_getenv_p("SCI_DEBUG")) {
          cerr << it->first << " size: " << it->second->numItems() << "\n";
        }
        it->second->send(event);
      }
    }
    mboxes_lock_.unlock();
  } while (!done);
}

} // namespace SCIRun
