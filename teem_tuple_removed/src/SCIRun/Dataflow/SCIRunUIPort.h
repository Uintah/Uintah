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
 *  SCIRunUIPort.h: CCA-style Interface to old TCL interfaces
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 2002
 *
 */

#ifndef SCIRun_Dataflow_SCIRunUIPort_h
#define SCIRun_Dataflow_SCIRunUIPort_h

#include <Core/CCA/spec/cca_sidl.h>

namespace SCIRun {
  class SCIRunComponentInstance;
  class SCIRunUIPort : public sci::cca::ports::UIPort {
  public:
    SCIRunUIPort(SCIRunComponentInstance* component);
    virtual ~SCIRunUIPort();

    virtual int ui();
  private:
    SCIRunComponentInstance* component;
    SCIRunUIPort(const SCIRunUIPort&);
    SCIRunUIPort& operator=(const SCIRunUIPort&);
  };
}

#endif
