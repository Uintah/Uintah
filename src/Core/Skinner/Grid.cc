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
//    File   : Grid.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:03:07 2006

#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Grid.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Core/Util/Assert.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>


namespace SCIRun {
  namespace Skinner {
    
    Grid::Grid(Variables *variables) :
      Parent(variables),
      rows_(variables,"rows"),
      cols_(variables,"cols"),
      cell_info_(),
      cell_width_(cols_(), AIR_NEG_INF),
      cell_height_(rows_(), AIR_NEG_INF)

    {
      REGISTER_CATCHER_TARGET(Grid::ReLayoutCells);
    }

    Grid::~Grid() {
      // Parent class deletes children for us
    }

    BaseTool::propagation_state_e
    Grid::process_event(event_handle_t event)
    {
      if (!visible_()) return STOP_E;
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() == WindowEvent::REDRAW_E) {
        ReLayoutCells(event);
      }

      const RectRegion &region = get_region();
      int rows = rows_();
      int cols = cols_();

      double usedy = 0.0;
      double unusedy = 0.0;
      for (int r = 0; r < rows; ++r)
        if (cell_height_[r] < 0.0)
          unusedy += 1.0;
        else if (cell_height_[r] > 1.0)
          usedy += cell_height_[r];

      vector<double> posy(rows+1, 0);      
      for (int r = 1; r <= rows; ++r) {
        double dy = cell_height_[r-1];
        if (dy < 0.0)
          dy = (region.height()-usedy)/unusedy;
        else if (dy < 1.0)
          dy = region.height()*dy;
        posy[r] = Clamp(posy[r-1] + dy, 0.0, region.height());
      }

      double usedx = 0.0;
      double unusedx = 0.0;
      for (int c = 0; c < cols; ++c)
        if (cell_width_[c] < 0.0)
          unusedx += 1.0;
        else if (cell_width_[c] > 1.0)
          usedx += cell_width_[c];

      vector<double> posx(cols+1, 0);

      for (int c = 1; c <= cols; ++c) {
        double dx = cell_width_[c-1];
        if (dx < 0.0)
          dx = (region.width() - usedx)/unusedx;
        else if (dx < 1.0)
          dx = region.width()*dx;

        posx[c] = Clamp(posx[c-1] + dx, 0.0, region.width());
      }

      for (unsigned int i = 0; i < cell_info_.size(); ++i) {
        CellInfo_t &cell = cell_info_[i];
        int r = cell.row_()-1;
        int c = cell.col_()-1;
        children_[i]->set_region(RectRegion(region.x1() + posx[c], 
                                            region.y2() - posy[r + 1], 
                                            region.x1() + posx[c + 1], 
                                            region.y2() - posy[r]));
        children_[i]->process_event(event);
      }

      return CONTINUE_E;
    }



    void
    Grid::set_children(const Drawables_t &children) {
      children_ = children;
      cell_info_.resize(children_.size());
      for (unsigned int i = 0; i < children_.size(); ++i) {        
        Variables *cvars = children_[i]->get_vars();
        CellInfo_t &cell = cell_info_[i];
        cell.row_ = Var<int>(cvars, "row");
        cell.col_ = Var<int>(cvars, "col");
        cell.width_ = Var<double>(cvars, "cell-width");
        cell.height_ = Var<double>(cvars, "cell-height");
      }
      
    }


    BaseTool::propagation_state_e
    Grid::ReLayoutCells(event_handle_t) {
      cell_width_ = vector<double>(cell_width_.size(), AIR_NEG_INF);
      cell_height_ = vector<double>(cell_height_.size(), AIR_NEG_INF);

      for (unsigned int i = 0; i < cell_info_.size(); ++i) {
        CellInfo_t &cell = cell_info_[i];
        int row = cell.row_()-1;
        int col = cell.col_()-1;
        ASSERT(row < rows_());
        ASSERT(col < cols_());
        
        if (cell.width_.exists()) {
          cell_width_[col] = Max(cell_width_[col], cell.width_());
        }
        
        if (cell.height_.exists()) {
          cell_height_[row] = Max(cell_height_[row], cell.height_());
        }
      }
      return STOP_E;
    }
      
    Drawable *
    Grid::maker(Variables *vars) 
    {
      return new Grid(vars);
    }
  }
}
