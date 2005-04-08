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

class GeomFastQuads : public GeomObj {
protected:
  vector<float> points_;
  vector<unsigned char> colors_;
  vector<float> indices_;
  vector<float> normals_;
  MaterialHandle material_;

public:
  GeomFastQuads();
  GeomFastQuads(const GeomFastQuads&);
  virtual ~GeomFastQuads();
  virtual GeomObj* clone();

  int size(void);
  void add(const Point &p0, const Point &p1,
	   const Point &p2, const Point &p3);
  void add(const Point &p0, const Vector &n0,
	   const Point &p1, const Vector &n1,
	   const Point &p2, const Vector &n2,
	   const Point &p3, const Vector &n3);
  void add(const Point &p0, const MaterialHandle &m0,
	   const Point &p1, const MaterialHandle &m1,
	   const Point &p2, const MaterialHandle &m2,
	   const Point &p3, const MaterialHandle &m3);
  void add(const Point &p0, double cindex0,
	   const Point &p1, double cindex1,
	   const Point &p2, double cindex2,
	   const Point &p3, double cindex3);
  void add(const Point &p0, const Vector &n0, const MaterialHandle &m0,
	   const Point &p1, const Vector &n1, const MaterialHandle &m1,
	   const Point &p2, const Vector &n2, const MaterialHandle &m2,
	   const Point &p3, const Vector &n3, const MaterialHandle &m3);
  void add(const Point &p0, const Vector &n0, double cindex0,
	   const Point &p1, const Vector &n1, double cindex1,
	   const Point &p2, const Vector &n2, double cindex2,
	   const Point &p3, const Vector &n3, double cindex3);


  virtual void get_bounds(BBox& bb);

#ifdef SCI_OPENGL
  virtual void draw(DrawInfoOpenGL*, Material*, double time);
#endif

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};



class GeomTranspQuads : public GeomFastQuads
{
protected:
  vector<unsigned int> xlist_;
  vector<unsigned int> ylist_;
  vector<unsigned int> zlist_;
  bool xreverse_;
  bool yreverse_;
  bool zreverse_;

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

