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
    
    Grid::Grid(Variables *variables, int rows, int cols) :
      Parent(variables),
      cells_(rows, vector<Drawable *>(cols)),
      col_width_(cols+1, AIR_NEG_INF),
      row_height_(rows+1, AIR_NEG_INF)    
    {
      REGISTER_CATCHER_TARGET(Grid::ReLayoutCells);
    }

    Grid::~Grid() {
      // Parent class deletes children for us
    }

    BaseTool::propagation_state_e
    Grid::process_event(event_handle_t event)
    {
      WindowEvent *window = dynamic_cast<WindowEvent *>(event.get_rep());
      if (window && window->get_window_state() & WindowEvent::REDRAW_E) {
        ReLayoutCells(event);
      }
      const RectRegion &region = get_region();
      unsigned int rows = cells_.size();
      double usedy = 0.0;
      double unusedy = 0.0;
      for (unsigned int r = 0; r < rows; ++r)
        if (row_height_[r] < 0.0)
          unusedy += 1.0;
        else if (row_height_[r] > 1.0)
          usedy += row_height_[r];

      vector<double> posy(rows+1, 0);      
      for (unsigned int r = 1; r <= rows; ++r) {
        double dy = row_height_[r-1];
        if (dy < 0.0)
          dy = (region.height()-usedy)/unusedy;
        else if (dy < 1.0)
          dy = region.height()*dy;
        posy[r] = Clamp(posy[r-1] + dy, 0.0, region.height());
      }

      unsigned int cols = cells_[0].size();
      double usedx = 0.0;
      double unusedx = 0.0;
      for (unsigned int c = 0; c < cols; ++c)
        if (col_width_[c] < 0.0)
          unusedx += 1.0;
        else if (col_width_[c] > 1.0)
          usedx += col_width_[c];

      vector<double> posx(cols+1, 0);

      for (unsigned int c = 1; c <= cols; ++c) {
        double dx = col_width_[c-1];
        if (dx < 0.0)
          dx = (region.width() - usedx)/unusedx;
        else if (dx < 1.0)
          dx = region.width()*dx;

        posx[c] = Clamp(posx[c-1] + dx, 0.0, region.width());
      }

      for (unsigned int r = 0; r < rows; ++r) {
        for (unsigned int c = 0; c < cols; ++c) {
          if (!cells_[r][c]) continue;
          cells_[r][c]->set_region(RectRegion(region.x1() + posx[c], 
                                              region.y2() - posy[r + 1], 
                                              region.x1() + posx[c + 1], 
                                              region.y2() - posy[r]));
          cells_[r][c]->process_event(event);
        }
      }

      return CONTINUE_E;
    }



    void
    Grid::set_children(const Drawables_t &children) {
      children_ = children;
      for (Drawables_t::const_iterator iter = children.begin(); 
           iter != children.end(); ++iter) {
        Variables *cvars = (*iter)->get_vars();
        ASSERT(cvars);
        int row = 1;
        cvars->maybe_get_int("row", row);
        
        int col = 1;
        cvars->maybe_get_int("col", col);
        
        double width = AIR_NEG_INF;
        cvars->maybe_get_double("cell-width", width);
        
        double height = AIR_NEG_INF;
        cvars->maybe_get_double("cell-height", height);
        
        
        set_cell(row, col, *iter, width, height);
      }
    }



    void
    Grid::set_cell(int row, int col, Drawable *obj, 
                   double width, double height) 
    {
      //      const unsigned int rows = cells_.size();
      //      const unsigned int cols = cells_[0].size();
      //      row = cells_.size() - row;
      row--;
      col--;//cells_[row].size() - col;

      if (cells_[row][col]) {
        throw (get_id() + "Row: " + to_string(row) + 
               " Col: " + to_string(col) + " already occupied by: " +
               cells_[row][col]->get_id());
      }

      cells_[row][col] = obj;

      col_width_[col] = Max(col_width_[col], width);
      row_height_[row] = Max(row_height_[row], height);


    }


    BaseTool::propagation_state_e
    Grid::ReLayoutCells(event_handle_t) {
#if 1
      col_width_ = vector<double>(col_width_.size(), AIR_NEG_INF);
      row_height_ = vector<double>(row_height_.size(), AIR_NEG_INF);
      for (Drawables_t::const_iterator iter = children_.begin(); 
           iter != children_.end(); ++iter) {
        Variables *cvars = (*iter)->get_vars();
        ASSERT(cvars);
        int row = 1;
        cvars->maybe_get_int("row", row);
        
        int col = 1;
        cvars->maybe_get_int("col", col);
        
        double width = AIR_NEG_INF;
        cvars->maybe_get_double("cell-width", width);
        
        double height = AIR_NEG_INF;
        cvars->maybe_get_double("cell-height", height);

        row--;
        col--;
        col_width_[col] = Max(col_width_[col], width);
        row_height_[row] = Max(row_height_[row], height);
      }
#endif
      return STOP_E;
    }
      


    Drawable *
    Grid::maker(Variables *vars) 
    {
      int rows = 1;
      vars->maybe_get_int("rows", rows);

      int cols = 1;
      vars->maybe_get_int("cols", cols);

      return new Grid(vars, rows, cols);      
    }
  }
}
