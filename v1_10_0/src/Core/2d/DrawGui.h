/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rightsget_iports(name Reserved.
*/


/*
 *  DrawGui.h: 
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Aug 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_DrawGui_h
#define SCI_DrawGui_h 

#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/2d/DrawObj.h>

namespace SCIRun {
  
class SCICORESHARE DrawGui : public TclObj, public DrawObj {
protected:
  string menu_, tb_, ui_;

public:
  DrawGui( GuiInterface* gui, const string &name, const string &script );
  virtual ~DrawGui();

  virtual void set_windows( const string &menu, const string &tb,
			    const string &ui, const string &ogl="" );

  virtual void redraw() { draw(); }
};

} // namespace SCIRun

#endif // SCI_DrawGui_h
