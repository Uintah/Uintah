/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  SearchGrid.h: Specialized compact regular mesh used for searching.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2004
 *
 *
 */

#ifndef SCI_project_SearchGrid_h
#define SCI_project_SearchGrid_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <list>
#include <vector>

#include <Core/Datatypes/share.h>

namespace SCIRun {

typedef unsigned int under_type;

class SCISHARE SearchGridBase
{
public:

  SearchGridBase(int x, int y, int z,
		 const Point &min, const Point &max);

  SearchGridBase(int x, int y, int z, const Transform &t);

  virtual ~SearchGridBase();
  
  //! get the mesh statistics
  int get_ni() const { return ni_; }
  int get_nj() const { return nj_; }
  int get_nk() const { return nk_; }

  //virtual BBox get_bounding_box() const;
  void transform(const Transform &t);
  const Transform &get_transform() const { return transform_; }
  void get_canonical_transform(Transform &t);
  Transform &set_transform(const Transform &trans) 
  { transform_ = trans; return transform_; }

  bool locate(int &i, int &j, int &k, const Point &) const;
  void unsafe_locate(int &i, int &j, int &k, const Point &) const;

protected:

  unsigned int linearize(int i, int j, int k) const
  {
    // k inner loops
    return ((i * nj_) + j) * nk_ + k;
  }

  int ni_, nj_, nk_;

  Transform transform_;
};


class SCISHARE SearchGridConstructor : public Datatype, public SearchGridBase
{
  friend class SearchGrid;
  
public:
  SearchGridConstructor(int x, int y, int z,
			const Point &min, const Point &max);
  virtual ~SearchGridConstructor();

  void insert(under_type val, const BBox &bbox);
  void remove(under_type val, const BBox &bbox);
  bool lookup(const std::list<under_type> *&candidates, const Point &p) const;
  void lookup_ijk(const std::list<under_type> *&candidates,
                  int i, int j, int k) const;
  double min_distance_squared(const Point &p, int i, int j, int k) const;

  virtual void io(Piostream&) {}

protected:
  std::vector<std::list<under_type> > bin_;
  unsigned int size_;
};



class SCISHARE SearchGrid : public Datatype, public SearchGridBase
{
public:
  SearchGrid(const SearchGridConstructor &c);
  SearchGrid(const SearchGrid &c);
  virtual ~SearchGrid();

  bool lookup(under_type **begin, under_type **end, const Point &p) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

protected:
  SearchGrid();

  std::vector<unsigned int> accum_;
  under_type *vals_;
  unsigned int vals_size_;

  // Returns a SearchGrid
  static Persistent *maker() { return new SearchGrid(); }
};


} // namespace SCIRun

#endif // SCI_project_SearchGrid_h
