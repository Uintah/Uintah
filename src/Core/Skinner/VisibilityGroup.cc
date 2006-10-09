//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : VisibilityGroup.cc
//    Author : McKay Davis
//    Date   : Tue Oct  3 23:17:07 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/VisibilityGroup.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>

namespace SCIRun {
  namespace Skinner {
    VisibilityGroup::VisibilityGroup(Variables *variables) :
      Parent(variables)
    {
      REGISTER_CATCHER_TARGET(VisibilityGroup::show_VisibleItem);
      REGISTER_CATCHER_TARGET(VisibilityGroup::VisibleItem_Maker);
    }



    BaseTool::propagation_state_e
    VisibilityGroup::VisibleItem_Maker(event_handle_t maker_signal)
    {
      VisibleItem *child = 
        construct_child_from_maker_signal<VisibleItem>(maker_signal);
      visible_items_.push_back(child);
      return STOP_E;
    }


    VisibilityGroup::~VisibilityGroup()
    {
    }

    MinMax
    VisibilityGroup::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }

    BaseTool::propagation_state_e
    VisibilityGroup::show_VisibleItem(event_handle_t event) {
      Signal *signal = dynamic_cast<Signal *>(event.get_rep());
      string id = "/"+signal->get_vars()->get_string("id");
      string group = signal->get_vars()->get_string("group");
      for (int i = 0; i < visible_items_.size(); ++i) {
        if (group == visible_items_[i]->get_vars()->get_string("group")) {
          bool wasvisible = visible_items_[i]->visible_();
          bool isvisible = ends_with(visible_items_[i]->get_id(), id);
          if (wasvisible && !isvisible) {
            visible_items_[i]->throw_made_invisible();
            visible_items_[i]->visible_ = false;
          } else if (!wasvisible && isvisible) {
            visible_items_[i]->visible_ = true;
            visible_items_[i]->throw_made_visible();
          }
        }
      }
      return CONTINUE_E;
    }        
    



    VisibleItem::VisibleItem(Variables *variables) :
      Parent(variables)
    {
    }

    VisibleItem::~VisibleItem()
    {
    }

    void
    VisibleItem::throw_made_visible() {
      throw_signal("VisibleItem::made_visible");
    }

    void
    VisibleItem::throw_made_invisible() {
      throw_signal("VisibleItem::made_invisible");
    }



    int
    VisibleItem::get_signal_id(const string &signalname) const {
      if (signalname == "VisibleItem::made_visible") return 1;
      if (signalname == "VisibleItem::made_invisible") return 2;
      return 0;
    }


    MinMax
    VisibleItem::get_minmax(unsigned int ltype)
    {
      return SPRING_MINMAX;
    }


  }
}
