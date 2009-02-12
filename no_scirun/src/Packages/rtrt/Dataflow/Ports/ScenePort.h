/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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

#include <Dataflow/Network/Ports/SimplePort.h>
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
