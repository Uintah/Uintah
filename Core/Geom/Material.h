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
 *  Material.h:  Material Properities for Geometry
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Material_h
#define SCI_Geom_Material_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomContainer.h>

namespace SCIRun {

class SCICORESHARE Material : public Persistent {
public:
  int ref_cnt;
  Mutex lock;

  Color ambient;
  Color diffuse;
  Color specular;
  double shininess;
  Color emission;
  double reflectivity;
  double transparency;
  double refraction_index;

  Material();
  Material(const Color&, const Color&, const Color&, double);
  Material(const Color&);
  Material(const Material&);
  Material& operator=(const Material&);
  ~Material();
  Material* clone();

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

typedef LockingHandle<Material> MaterialHandle;

class SCICORESHARE GeomMaterial : public GeomContainer {
  MaterialHandle matl;
public:
  GeomMaterial(GeomHandle, const MaterialHandle&);
  GeomMaterial(const GeomMaterial&);
  void setMaterial(const MaterialHandle&);
  MaterialHandle getMaterial();
  virtual ~GeomMaterial();
  virtual GeomObj* clone();

  // For OpenGL
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif
