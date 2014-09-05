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
 *  GuiView.h   Structure that provides for easy access of view information.
 *              The view information is interactively provided by the user.
 *
 *  Written by:
 *   Steven Parker
 *   Department of Computer Science
 *   University of Utah
 *
 *   separated from the Viewer code by me (Aleksandra)
 *   in May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_project_GuiView_h
#define SCI_project_GuiView_h 1

#include <Core/share/share.h>

#include <Core/Geom/View.h>
#include <Core/Geom/GuiGeom.h>
#include <Core/Containers/String.h>
#include <Core/GuiInterface/GuiVar.h>

namespace SCIRun {


class SCICORESHARE GuiView : public GuiVar {
    GuiPoint eyep;
    GuiPoint lookat;
    GuiVector up;
    GuiDouble fov;
    GuiVector eyep_offset;
public:
    GuiView(const clString& name, const clString& id, TCL* tcl);
    ~GuiView();
    GuiView(const GuiView&);

    virtual void reset();
    View get();
    void set(const View&);
    virtual void emit(std::ostream& out, clString& midx);
};

class SCICORESHARE GuiExtendedView : public GuiVar {
    GuiPoint eyep;
    GuiPoint lookat;
    GuiVector up;
    GuiDouble fov;
    GuiVector eyep_offset;

    GuiInt   xres;
    GuiInt   yres;

public:
    GuiColor bg;
    GuiExtendedView(const clString& name, const clString& id, TCL* tcl);
    ~GuiExtendedView();
    GuiExtendedView(const GuiExtendedView&);

    virtual void reset();
    ExtendedView get();
    void set(const ExtendedView&);
    virtual void emit(std::ostream& out, clString& midx);
  };

} // End namespace SCIRun


#endif
