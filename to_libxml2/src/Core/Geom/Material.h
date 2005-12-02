/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

class Material : public Persistent {
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

class GeomMaterial : public GeomContainer {
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
