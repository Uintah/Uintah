/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
  Permission is hereby granted, free of charge, to any person obtaining a
  copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation
  the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the
  Software is furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included
  in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THE SOFTWARE.
*/


/*
 *  TxtModule.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#ifndef CCA_Components_TxtBuilder_TxtModule_h
#define CCA_Components_TxtBuilder_TxtModule_h

#include <menu.h>
#include <panel.h>
#include <string>
#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/TxtBuilder/TxtConnection.h>


using namespace std;
namespace SCIRun {

  class PopMenu;

  class TxtModule{

  public:
    TxtModule(const sci::cca::ComponentID::pointer &cid,  const std::string& type);
    ~TxtModule();
    void draw();
    void show(int lines, int cols);
    void pop_menu();
    void show_ports();
    void hide();
    void reverse();
    void mv_to(int lines, int cols);
    void mv_up();
    void mv_down();
    void mv_left();
    void mv_right();
    int getCols();
    int getLines();
    std::string getName();
    std::string getType();

    sci::cca::ComponentID::pointer getCID();
    Rect rect();
  private:
    sci::cca::ComponentID::pointer cid;
    int lines, cols, width, height;
    string name;
    string type;
    bool is_highlight;
    WINDOW *win;
    PANEL *panel;
    PopMenu *popmenu;
    PopMenu *portmenu;
  };


} //namespace SCIRun
#endif
