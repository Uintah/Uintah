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
 *  Light.h: Base class for light sources
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Light_h
#define SCI_Geom_Light_h 1

#include <Core/share/share.h>

#ifndef _WIN32
#include <sci_config.h>
#endif
#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <Core/Persistent/Persistent.h>

namespace SCIRun {

class Point;
class Vector;
class Color;
class GeomObj;
class OcclusionData;
class View;
struct DrawInfoOpenGL;


class SCICORESHARE Light : public Persistent {
protected:
  Light(const string& name, bool on = true, bool tranformed = true);
    
public:
  int ref_cnt;
  Mutex &lock;
  string name;
  bool on;
  bool transformed;  // defaults to true meaning you do want the light 
                     // to be transformed by the current modelview matrix.  
                     // Set to false for headlights and directional lights 
                     // fixed in the viewing hemisphere.
  virtual ~Light();
  virtual void io(Piostream&);

  friend SCICORESHARE void Pio( Piostream&, Light*& );

  void opengl_reset_light( int i );

  static PersistentTypeID type_id;
#ifdef SCI_OPENGL
  virtual void opengl_setup(const View& view, DrawInfoOpenGL*, int& idx)=0;
#endif
};

typedef LockingHandle<Light> LightHandle;

} // End namespace SCIRun


#endif /* SCI_Geom_Light_h */

