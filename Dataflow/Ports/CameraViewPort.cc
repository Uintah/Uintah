
/*
 *  CameraViewPort.cc
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   August 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Ports/CameraViewPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

    
    extern "C" {
      PSECORESHARE IPort* make_CameraViewIPort(Module* module,
					       const clString& name) {
	return scinew SimpleIPort<CameraViewHandle>(module,name);
      }
      PSECORESHARE OPort* make_CameraViewOPort(Module* module,
					       const clString& name) {
	return scinew SimpleOPort<CameraViewHandle>(module,name);
      }
    }

    template<> clString SimpleIPort<CameraViewHandle>::port_type("CameraView");
    template<> clString SimpleIPort<CameraViewHandle>::port_color("chocolate1");

} // End namespace SCIRun
