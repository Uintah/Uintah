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
//    File   : VisibilityGroup.h
//    Author : McKay Davis
//    Date   : Tue Oct  3 23:16:26 2006

#ifndef SKINNER_VISIBILITYGROUP_H
#define SKINNER_VISIBILITYGROUP_H

#include <Core/Skinner/Parent.h>

namespace SCIRun {
  namespace Skinner {
    class VisibleItem;
    

    class VisibilityGroup : public Parent {
    public:
      VisibilityGroup (Variables *);
      virtual ~VisibilityGroup();
      virtual MinMax                    get_minmax(unsigned int);
    private:
      CatcherFunction_t                 VisibleItem_Maker;
      CatcherFunction_t                 show_VisibleItem;
      vector<VisibleItem*>              visible_items_;
    };

    class VisibleItem : public Parent {
    public:
      VisibleItem (Variables *);
      virtual ~VisibleItem();
      virtual MinMax                    get_minmax(unsigned int);
      virtual int                       get_signal_id(const string &) const;
    private:
      friend class VisibilityGroup;
      void                              throw_made_visible();
      void                              throw_made_invisible();
    };


  }
}

#endif
