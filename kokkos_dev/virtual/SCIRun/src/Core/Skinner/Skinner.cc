//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//  
//    File   : Skinner.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:01:42 2006

#include <Core/Skinner/Skinner.h>
#include <Core/Skinner/XMLIO.h>

#include <Core/Skinner/Animation.h>
#include <Core/Skinner/Box.h>
#include <Core/Skinner/Collection.h>
#include <Core/Skinner/ColorMap1D.h>
#include <Core/Skinner/Frame.h>
#include <Core/Skinner/Window.h>
#include <Core/Skinner/Gradient.h>
#include <Core/Skinner/Grid.h>
#include <Core/Skinner/SceneGraph.h>
#include <Core/Skinner/Texture.h>
#include <Core/Skinner/TextEntry.h>
#include <Core/Skinner/Root.h>
#include <Core/Skinner/Layout.h>
#include <Core/Events/Tools/FilterRedrawEventsTool.h>
#include <Core/Util/Environment.h>
#include <Core/Util/FileUtils.h>
#include <iostream>


using std::cerr;
using std::endl;

#define DEBUG 1

namespace SCIRun {
  namespace Skinner {
    static Root *skinner_root_ = 0;
    
    bool
    init_skinner() {
      XMLIO::register_maker<Animation>();
      XMLIO::register_maker<Box>();
      XMLIO::register_maker<Collection>();
      XMLIO::register_maker<Colormap1D>();
      XMLIO::register_maker<Frame>();
      XMLIO::register_maker<Gradient>();
      XMLIO::register_maker<Grid>();
      XMLIO::register_maker<Layout>();
      XMLIO::register_maker<SceneGraph>();
      XMLIO::register_maker<TextEntry>();
      XMLIO::register_maker<Texture>();

      return true;
    }

    Drawable *
    load_skin(const string &filename) {
#if !defined(DEBUG)
      try {
#endif
        init_skinner();  
        Root *root = Skinner::XMLIO::load(filename, skinner_root_);
        if (root) { 
          skinner_root_ = root;
        }
        EventManager::add_event(new WindowEvent(WindowEvent::REDRAW_E));
        root->spawn_redraw_threads();
#if !defined(DEBUG)
      } catch (const string &error) {
        cerr << "Skinner Error: " << error << std::endl;
        return 0;
      } catch (const char *&error) {
        cerr << "Skinner Error: " << error << std::endl;
        return 0;
      } catch (...) {
        cerr << "UNKNOWN Skinner Error" << std::endl;
        return 0;
      }
#endif

      return skinner_root_;
    }


    bool
    load_default_skin() {
      string default_skin = sci_getenv("SCIRUN_OBJDIR")+string("data");
      string skinner_path = default_skin;
      const char *path_ptr = sci_getenv("SKINNER_PATH");
      if (path_ptr) {
        skinner_path = string(path_ptr) + ":" + default_skin;
      }
      sci_putenv("SKINNER_PATH", skinner_path);
      sci_putenv("SCIRUN_FONT_PATH",skinner_path);
      const char *fn = sci_getenv("SKINNER_FILE");
      string filename = (fn ? fn : "main.skin");
      
      string path = findFileInPath(filename, skinner_path);
      if (path.empty()) {
        std::cerr << "Cannot find main.skin in SKINNER_PATH.\n";
        std::cerr << "SKINNER_PATH=" << skinner_path << std::endl;;
        return 0;
      }

      return Skinner::load_skin(path+filename) ? true : false;
    }



    ThrottledRunnableToolManager::ThrottledRunnableToolManager
    (const string &name, double hertz) :
      ThrottledRunnable(hertz),
      mailbox_(SCIRun::EventManager::register_event_messages(name)),      
      tool_manager_(new ToolManager(name)),
      name_(name),
      tm_lock_(new Mutex(name.c_str()))
    {
    }

    
    ThrottledRunnableToolManager::~ThrottledRunnableToolManager() {
      delete tool_manager_;
      delete tm_lock_;
      SCIRun::EventManager::unregister_mailbox(mailbox_);

    }



    void
    ThrottledRunnableToolManager::add_tool(tool_handle_t tool, unsigned int p)
    {
      ASSERT(tool_manager_);
      tm_lock_->lock();
      tool_manager_->add_tool(tool, p);
      tm_lock_->unlock();
    }
    

    bool
    ThrottledRunnableToolManager::iterate()
    {
      event_handle_t event = 0;
      while (mailbox_->tryReceive(event)) {
        tm_lock_->lock();
        tool_manager_->propagate_event(event);
        tm_lock_->unlock();
        // Quit this throttled runnable if QuitEvent type received
        if (dynamic_cast<QuitEvent *>(event.get_rep())) {
          return false;
        }
      }
      return true;
    }



    ThrottledRedraw::ThrottledRedraw(Drawable *drawable, double hertz) :
      ThrottledRunnable(hertz),
      drawable_(drawable),
      redraw_(false),
      mailbox_(EventManager::register_event_messages(drawable_->get_id()))
    {
      
    }
    
    ThrottledRedraw::~ThrottledRedraw() {
      EventManager::unregister_mailbox(mailbox_);
    }
    
    
    bool
    ThrottledRedraw::iterate()
    {
      event_handle_t event = 0;
      while (mailbox_->tryReceive(event)) {
        if (!redraw_) {
          WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
          if (window && window->get_window_state() & WindowEvent::REDRAW_E) {
            redraw_ = true;
          }
        }
      }
      //      event = 0;
      if (redraw_) {
        WindowEvent *window_event = new WindowEvent(WindowEvent::REDRAW_E);
        event_handle_t event = window_event;
        drawable_->process_event(event);
        redraw_ = false;
      }
      return true;
    }
  }
}
    
