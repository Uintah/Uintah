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
//    File   : FrameBufferPickTool.cc
//    Author : Martin Cole
//    Date   : Wed Jul 19 10:52:18 2006

#include <Core/Events/Tools/FrameBufferPickTool.h>
#include <Core/Events/keysyms.h>
#include <sstream>

namespace SCIRun {

FrameBufferPickTool::FrameBufferPickTool(string name, FBInterface *f) :
  PointerTool(name),
  fbi_(f)
{
}

FrameBufferPickTool::~FrameBufferPickTool()
{
  delete fbi_;
}

BaseTool::propagation_state_e
FrameBufferPickTool::pointer_down(int which, int x, int y, 
				  unsigned int modifiers, int time)
{
  if (which == 1) { 
    //! left mouse
    add_pick_to_selection(x, y);
  } else if (which == 3) { 
    //! right mouse
    remove_pick_from_selection(x, y);
  } else {
    return CONTINUE_E;
  }
  return STOP_E;
}

BaseTool::propagation_state_e
FrameBufferPickTool::pointer_motion(int which, int x, int y, 
				    unsigned int mod, int time)
{
  //pointer_down(which, x, y, mod, time);
  return STOP_E;
}

BaseTool::propagation_state_e
FrameBufferPickTool::pointer_up(int which, int x, int y, 
				unsigned int, int time)
{
  //cerr << "FrameBufferPickTool::pointer_up" << endl;
  return STOP_E;
}

inline
void 
rgba2idx(unsigned int &idx, unsigned char r, unsigned char g, 
	 unsigned char b, unsigned char a)
{
  // dont use alpha at all just rgb.
  idx = (r << 16) | (g << 8) | b;
}

bool
FrameBufferPickTool::get_index_at_selection(int x, int y, unsigned int &idx) 
{
  fbi_->do_pick_draw();
  unsigned int sz = fbi_->width() * fbi_->height() * 4;
  if (fbi_->img().size() != sz) {
    cerr << "ERROR: fbpick_image is invalid" << endl;
    return false;
  }
  y = fbi_->height() - y - 1;
  int off = y * fbi_->width() * 4 + x * 4;
  unsigned char r,g,b,a;
  r = fbi_->img()[off];
  g = fbi_->img()[off + 1];
  b = fbi_->img()[off + 2];
  a = fbi_->img()[off + 3];
  //  cerr << "rgba at x=" << x << " y=" << y << " :" 
  //     << (int)r << "," << (int)g << "," << (int)b << "," << (int)a << endl; 

  rgba2idx(idx, r, g, b, a); // the index is drawn + 1 so that we can 
                             // distinguish the element 0 from and empty pick.

//   if (idx == 0) return false;
//   idx--;
  cerr << "idx at x=" << x << " y=" << y << " :" << idx << endl;
  return true;
}

void
FrameBufferPickTool::add_pick_to_selection(int x, int y)
{
  unsigned int idx;
  if (get_index_at_selection(x, y, idx)) {
    fbi_->add_selection(idx);
  }
}

void
FrameBufferPickTool::remove_pick_from_selection(int x, int y)
{
  unsigned int idx;
  if (get_index_at_selection(x, y, idx)) {
    fbi_->remove_selection(idx);
  }
}

BaseTool::propagation_state_e 
FrameBufferPickTool::process_event(event_handle_t e)
{
  cerr << "FBPT process_event" << endl;
  return CONTINUE_E;
}
} // namespace SCIRun
