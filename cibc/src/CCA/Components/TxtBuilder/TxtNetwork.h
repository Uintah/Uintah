/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
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
 *  TxtNetwork.h: 
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   September 2005
 *
 */

#ifndef CCA_Components_TxtBuilder_TxtNetwork_h
#define CCA_Components_TxtBuilder_TxtNetwork_h

#include <curses.h>
#include <menu.h>
#include <vector>
#include <Core/CCA/spec/cca_sidl.h>


namespace SCIRun {

  class DropMenu;
  class TxtModule;

  class TxtNetwork{

  public:
    TxtNetwork();
    void setup(WINDOW* win_main);
    int enter();
    static void addModule(TxtModule*);
    static void addModule(TxtModule*, int row, int col);
    static void delModule(TxtModule*);
    static void delModule(sci::cca::ComponentID::pointer cid);
    static void clear();
    static void drawConnections();
    static std::vector<TxtModule*> getModuleList();
    static std::string getFilename();
    static void setFilename(const std::string& filename);
    static void writeFile();
    static void loadFile();
  private:
    static TxtModule* active_module;
    static std::vector<TxtModule*> module_list;
    static int location;
    static WINDOW* win;
    static std::string filename;
  };

} //namespace SCIRun

#endif
