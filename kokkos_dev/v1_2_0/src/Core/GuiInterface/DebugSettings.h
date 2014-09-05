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
  University of Utah. All Rights Reserved.
*/


/*
 *  DebugSettings.h: Debug settings visualizer
 *
 *  Written by:
 *   James T. Purciful
 *   Department of Computer Science
 *   University of Utah
 *   Oct. 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_DebugSettings_h
#define SCI_project_DebugSettings_h 1

#include <Core/Containers/Array1.h>
#include <Core/GuiInterface/TCL.h>

namespace SCIRun {


class GuiVarintp;

class SCICORESHARE DebugSettings : public TCL {
    Array1<GuiVarintp*> variables;
public:
    DebugSettings();
    ~DebugSettings();

    void init_tcl();
    virtual void tcl_command(TCLArgs&, void*);
};

} // End namespace SCIRun


#endif
