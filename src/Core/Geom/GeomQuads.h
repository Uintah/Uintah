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
 *  GeomQuads.h: Quads
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   May 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef SCI_Geom_Quads_h
#define SCI_Geom_Quads_h 1

#include <Core/Geom/GeomObj.h>
#include <Core/Geom/Material.h>

namespace SCIRun {

class SCICORESHARE GeomFastQuads : public GeomObj {
protected:
  vector<float> points_;
  vector<unsigned char> colors_;
  vector<float> normals_;
  MaterialHandle material_;

public:
  GeomFastQuads();
  GeomFastQuads(const GeomFastQuads&);
  virtual ~GeomFastQuads();
  virtual GeomObj* clone();

  int size(void);
  void add(const Point&, const MaterialHandle&,
	   const Point&, const MaterialHandle&,
	   const Point&, const MaterialHandle&,
	   const Point&, const MaterialHandle&);
  void add(const Point&, const Vector&, const MaterialHandle&,
	   const Point&, const Vector&, const MaterialHandle&,
	   const Point&, const Vector&, const MaterialHandle&,
	   const Point&, const Vector&, const MaterialHandle&);

  virtual void get_bounds(BBox& bb);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  virtual bool saveobj(std::ostream&, const string& format, GeomSave*);
};



class SCICORESHARE GeomTranspQuads : public GeomFastQuads
{
protected:
  vector<unsigned int> xlist_;
  vector<unsigned int> ylist_;
  vector<unsigned int> zlist_;

public:
  GeomTranspQuads();
  GeomTranspQuads(const GeomTranspQuads&);
  virtual ~GeomTranspQuads();
  virtual GeomObj* clone();

  void SortPolys();

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun


#endif /* SCI_Geom_Quads_h */

