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
 *  MessageTypes.h: enum definitions for Message Types...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MessageTypes_h
#define SCI_project_MessageTypes_h 1

#include <Dataflow/share/share.h>

namespace SCIRun {

class PSECORESHARE MessageTypes {
    MessageTypes();
public:
    void dummy(); // Keeps g++ happy...
    enum MessageType {
	DoCallback,
	DoDBCallback,
	ExecuteModule,
	TriggerPort,
	ReSchedule,
	MultiSend,
	GoAway,

	MUIDispatch,

	GeometryInit,
	GeometryDetach,
	GeometryAddObj,
	GeometryDelObj,
	GeometryAddLight,
	GeometryDelLight,
	GeometryDelAll,
	GeometryFlush,
	GeometryFlushViews,
	GeometryGetData,
	GeometryGetNViewWindows,
	GeometrySetView,

	GeometryPick,
	GeometryRelease,
	GeometryMoved,

	ViewWindowRedraw,
	ViewWindowMouse,
	ViewWindowDumpImage,
	ViewWindowDumpObjects,
	ViewWindowEditLight,

	AttachDialbox,
	DialMoved,

	TrackerMoved,

	ModuleGeneric1,
	ModuleGeneric2,
	ModuleGeneric3,
	ModuleGeneric4
	
    };
};

} // End namespace SCIRun


#endif
