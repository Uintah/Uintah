/*
 *  ScenePort.h - port for passing scene handles
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   September 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef __RTRT_SCENEPORT_H__
#define __RTRT_SCENEPORT_H__

#include <Packages/rtrt/Core/Scene.h>

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/Persistent.h>

namespace rtrt {

  using namespace SCIRun;

  class SceneContainer: public Datatype {
    Scene *scene;
  public:
    Scene* get_scene() { return scene; }
    void put_scene(Scene *_scene) { scene = _scene; }

    // Persistant representation
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
  };

  typedef LockingHandle<SceneContainer> SceneContainerHandle;

  typedef SimpleIPort<SceneContainerHandle> SceneIPort;
  typedef SimpleOPort<SceneContainerHandle> SceneOPort;

} // end namespace rtrt

#endif // __RTRT_SCENEPORT_H__
