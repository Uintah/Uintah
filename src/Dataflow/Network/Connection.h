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
 *  Connection.h: A Connection between two modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Connection_h
#define SCI_project_Connection_h 1

#include <Dataflow/share/share.h>
#include <string>

namespace SCIRun {
  class IPort;
  class Module;
  class OPort;

class PSECORESHARE Connection {
public:
  Connection(Module* omod, int oportno, Module* imod, int imodno,
	     const std::string &id);
  ~Connection();

  OPort* const oport;
  IPort* const iport;
  std::string id;

  void makeID();
};

} // End namespace SCIRun


#endif /* SCI_project_Connection_h */
