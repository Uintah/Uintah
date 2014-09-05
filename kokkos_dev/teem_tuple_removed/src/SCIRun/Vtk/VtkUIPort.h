/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is Vtk, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  VtkUIPort.h: CCA-style Interface to old TCL interfaces
 *
 *  Written by:
 *   Keming Zhang
 *   Department of Computer Science
 *   University of Utah
 *   January 2004
 *
 */

#ifndef SCIRun_Vtk_VtkUIPort_h
#define SCIRun_Vtk_VtkUIPort_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun{
  class VtkComponentInstance;
  class VtkUIPort : public sci::cca::ports::UIPort {
  public:
    VtkUIPort(VtkComponentInstance* ci);
    virtual ~VtkUIPort();

    virtual int ui();
  private:
    VtkComponentInstance* ci;
    VtkUIPort(const VtkUIPort&);
    VtkUIPort& operator=(const VtkUIPort&);
  };
}

#endif
