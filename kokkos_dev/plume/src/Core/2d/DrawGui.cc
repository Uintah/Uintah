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
 *  DrawGui.cc: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */


#include <Core/Malloc/Allocator.h>
#include <Core/2d/DrawGui.h>

namespace SCIRun {


  //PersistentTypeID DrawGui::type_id("DrawGUI", "DrawObj", make_DrawGui);

DrawGui::DrawGui(GuiInterface* gui, const string &name, const string& script)
  : TclObj(gui, script), DrawObj(name)
{
}

void 
DrawGui::set_windows( const string &menu, const string &tb,
		      const string &ui, const string &ogl )
{
  menu_ = menu;
  tb_ = tb;
  ui_ = ui;
  
  string space(" ");
  TclObj::set_window( id(), 
		      menu+" "+tb+" "+ui+" "+ogl );
}

DrawGui::~DrawGui()
{
}

} // namespace SCIRun

  
