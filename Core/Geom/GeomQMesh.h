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
 *  GeomQMesh.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

/*
 * This class does color and point per vertex
 * it stores everythning explicitly.
 * Peter-Pike Sloan
 */

#ifndef SCI_Geom_QMesh_h
#define SCI_Geom_QMesh_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {

class SCICORESHARE GeomQMesh : public GeomObj {
  Array1<float> pts;    // data points
  Array1<float> nrmls;  // normals for above (central differences???)
  Array1<Colorub> clrs; // colors

  int nrows,ncols; // 2d grid of pts...

public:
  GeomQMesh(int, int);
  GeomQMesh(const GeomQMesh&);
  
  virtual ~GeomQMesh();

  virtual GeomObj* clone();

  void add(int, int, Point&, Vector&, Color&); // adds point...
 
#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif
  virtual void get_bounds(BBox&);
  
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun



#endif
