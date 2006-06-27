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
#include <iostream>


namespace SCIRun {
  namespace Skinner {
    
    Grid::Grid(Variables *variables, int rows, int cols) :
      Drawable(variables),
      cells_(rows, vector<Drawable *>(cols)),
      col_width_(cols+1, AIR_NEG_INF),
      row_height_(rows+1, AIR_NEG_INF)    
    {
    }

    Grid::~Grid() {
      for (unsigned int r = 0; r < cells_.size(); ++r) {
        for (unsigned int c = 0; c < cells_[r].size(); ++c) {
          if (!cells_[r][c]) continue;
          delete cells_[r][c];
        }
      }
    }

    BaseTool::propagation_state_e
    Grid::process_event(event_handle_t event)
    {
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
          dy = (region_.height()-usedy)/unusedy;
        else if (dy < 1.0)
          dy = region_.height()*dy;
        posy[r] = Clamp(posy[r-1] + dy, 0.0, region_.height());
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
          dx = (region_.width() - usedx)/unusedx;
        else if (dx < 1.0)
          dx = region_.width()*dx;

        posx[c] = Clamp(posx[c-1] + dx, 0.0, region_.width());
      }

      for (unsigned int r = 0; r < rows; ++r) {
        for (unsigned int c = 0; c < cols; ++c) {
          if (!cells_[r][c]) continue;
          cells_[r][c]->region() = RectRegion(region_.x1() + posx[c], 
                                              region_.y2() - posy[r + 1], 
                                              region_.x1() + posx[c + 1], 
                                              region_.y2() - posy[r]);
          cells_[r][c]->process_event(event);
        }
      }

      return CONTINUE_E;
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
      if (cells_[row][col])
        std::cerr << std::endl;
      ASSERT(cells_[row][col] == 0);
      cells_[row][col] = obj;

      col_width_[col] = Max(col_width_[col], width);
      row_height_[row] = Max(row_height_[row], height);


    }


    Drawable *
    Grid::maker(Variables *vars, const Drawables_t &children, void *) 
    {
      int rows = 1;
      vars->maybe_get_int("rows", rows);

      int cols = 1;
      vars->maybe_get_int("cols", cols);

      Grid *grid = new Grid(vars, rows, cols);

      for (Drawables_t::const_iterator iter = children.begin(); 
           iter != children.end(); ++iter) {
        Variables *cvars = (*iter)->get_vars();
        if (cvars) {
          int row = 1;
          cvars->maybe_get_int("row", row);

          int col = 1;
          cvars->maybe_get_int("col", col);
          
          double width = AIR_NEG_INF;
          cvars->maybe_get_double("cell-width", width);

          double height = AIR_NEG_INF;
          cvars->maybe_get_double("cell-height", height);

          
          grid->set_cell(row, col, *iter, width, height);
        }
      }

      return grid;
      
    }
  }
}
