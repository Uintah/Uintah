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
//    File   : FloodfillTool.cc
//    Author : McKay Davis
//    Date   : Sat Oct 14 16:13:30 2006

#include <StandAlone/Apps/Painter/FloodfillTool.h>
#include <StandAlone/Apps/Painter/Painter.h>

namespace SCIRun {

FloodfillTool::FloodfillTool(Painter *painter) :
  BaseTool("Flood Fill"),
  PointerTool("Flood Fill"),
  painter_(painter),
  value_(0.0),
  min_(0.0),
  max_(0.0),
  start_pos_(0,0,0)
{
  painter_->create_undo_volume();
}


FloodfillTool::~FloodfillTool()
{
}


#if 0
int
FloodfillTool::do_event(Event &event) {
  if (!event.window_ || !painter_->current_volume_) 
    return FALLTHROUGH_E;

  if (event.type_ == Event::KEY_PRESS_E &&
      event.key_ == "q")
    return QUIT_E;


  if (event.type_ == Event::KEY_PRESS_E &&
      event.key_ == " ") {
    do_floodfill();
    return QUIT_E;
  }


  NrrdVolume *volume = painter_->current_volume_;
    
  if (event.type_ == Event::BUTTON_PRESS_E) {
    vector<int> index = volume->world_to_index(event.position_);
    if (!volume->index_valid(index))
      return FALLTHROUGH_E;
    if (event.button(1)) {
      min_ = volume->data_max_;
      max_ = volume->data_min_;
      start_pos_ = event.position_;
    }
    return HANDLED_E;
  }
  
  if (event.type_ == Event::MOTION_E) {

    vector<int> index = 
      volume->world_to_index(event.position_);
    if (!volume->index_valid(index)) 
      return FALLTHROUGH_E;
    
    double val;
    volume->get_value(index, val);
    
    if (event.button(1)) {
      min_ = Min(min_, val);
      max_ = Max(max_, val);
      cerr << "Min: " << min_ << "  Max: " << max_ << std::endl;
    }
    
    if (event.button(3)) {
      value_ = val;
      cerr << "value: " << value_ << std::endl;
    }
    painter_->redraw_all();
    return HANDLED_E;
  }
  
  return FALLTHROUGH_E;
}

#endif

int
FloodfillTool::draw(SliceWindow &)
{
  return 0;
}


void
FloodfillTool::do_floodfill()
{
  NrrdVolume *volume = painter_->current_volume_;
  vector<int> index = volume->world_to_index(start_pos_);
  if (!volume->index_valid(index)) 
    return;

  // Array to hold which indices to visit next
  vector<vector<int> > todo, oldtodo;

  // Push back the first seed point
  todo.push_back(index);

  // Allocated a nrrd to mark where the flood fill has visited
  NrrdDataHandle done_handle = new NrrdData();
  size_t size[NRRD_DIM_MAX];
  size[0] = volume->nrrd_handle_->nrrd_->axis[0].size;
  size[1] = volume->nrrd_handle_->nrrd_->axis[1].size;
  size[2] = volume->nrrd_handle_->nrrd_->axis[2].size;
  size[3] = volume->nrrd_handle_->nrrd_->axis[3].size;
  nrrdAlloc_nva(done_handle->nrrd_, nrrdTypeUChar, 4, size);


  // Set the visited nrrd to empty
  memset(done_handle->nrrd_->data, 0, 
         volume->nrrd_handle_->nrrd_->axis[0].size *
         volume->nrrd_handle_->nrrd_->axis[1].size *
         volume->nrrd_handle_->nrrd_->axis[2].size * 
         volume->nrrd_handle_->nrrd_->axis[3].size);
  int count  = 0;
  unsigned int axes = index.size();
  while (!todo.empty()) {
    ++count;
    if (!(count % 40)) {
      cerr << todo.size() << std::endl;
      painter_->set_all_slices_tex_dirty();
      painter_->redraw_all();
    }
      
    oldtodo = todo;
    todo.clear();
    for (unsigned int i = 0; i < oldtodo.size(); ++i)
      volume->set_value(oldtodo[i], value_);
    

    // For each axis
    for (unsigned int i = 0; i < oldtodo.size(); ++i) {
      for (unsigned int a = 1; a < axes; ++a) {
        // Visit the previous and next neighbor indices along this axis
        for (int dir = -1; dir < 2; dir +=2) {
          // Neighbor index starts as current index
          vector<int> neighbor_index = oldtodo[i];
          // Index adjusted in direction we're looking at along axis
          neighbor_index[a] = neighbor_index[a] + dir;
          // Bail if this index is outside the volume
          if (!volume->index_valid(neighbor_index)) continue;
          
          // Check to see if flood fill has already been here
          unsigned char visited;
          nrrd_get_value(done_handle->nrrd_, neighbor_index, visited);
          // Bail if the voxel has been visited
          if (visited) continue;
          
          // Now check to see if this pixel is a candidate to be filled
          double neighborval;
          volume->get_value(neighbor_index, neighborval);
          // Bail if the voxel is outside the flood fill range
          if (neighborval < min_ || neighborval > max_) continue;
          // Mark this voxel as visited
          nrrd_set_value(done_handle->nrrd_, neighbor_index, (unsigned char)1);
          
          todo.push_back(neighbor_index);
        }
      }
    }
  }

  painter_->set_all_slices_tex_dirty();
  painter_->redraw_all();
}

}  
