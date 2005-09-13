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
 *  SearchGrid.h: Specialized compact regular mesh used for searching.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2004
 *
 *  Copyright (C) 2004 SCI Group
 *
 */

#ifndef SCI_project_SearchGrid_h
#define SCI_project_SearchGrid_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <list>
#include <vector>

namespace SCIRun {

typedef unsigned int under_type;

class SearchGridBase
{
public:

  SearchGridBase(unsigned int x, unsigned int y, unsigned int z,
		 const Point &min, const Point &max);

  SearchGridBase(unsigned int x, unsigned int y, unsigned int z,
		 const Transform &t);

  virtual ~SearchGridBase() {}
  
  //! get the mesh statistics
  unsigned get_ni() const { return ni_; }
  unsigned get_nj() const { return nj_; }
  unsigned get_nk() const { return nk_; }

  //virtual BBox get_bounding_box() const;
  void transform(const Transform &t);
  const Transform &get_transform() const { return transform_; }
  void get_canonical_transform(Transform &t);
  Transform &set_transform(const Transform &trans) 
  { transform_ = trans; return transform_; }

protected:
  bool locate(unsigned int &i, unsigned int &j, unsigned int &k,
	      const Point &) const;
  void unsafe_locate(unsigned int &i, unsigned int &j, unsigned int &k,
		     const Point &) const;

  unsigned int linearize(unsigned int i, unsigned int j, unsigned int k) const
  {
    // k inner loops
    return ((i * nj_) + j) * nk_ + k;
  }

  unsigned int ni_, nj_, nk_;

  Transform transform_;
};


class SearchGridConstructor : public SearchGridBase
{
  friend class SearchGrid;
  
public:
  SearchGridConstructor(unsigned int x, unsigned int y, unsigned int z,
			const Point &min, const Point &max);

  void insert(under_type val, const BBox &bbox);
  
protected:
  std::vector<std::list<under_type> > bin_;
  unsigned int size_;
};



class SearchGrid : public Datatype, public SearchGridBase
{
public:
  SearchGrid(const SearchGridConstructor &c);
  virtual ~SearchGrid();

  bool lookup(under_type **begin, under_type **end, const Point &p) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  //static  const string type_name(int n = -1);
  //virtual const TypeDescription *get_type_description() const;

protected:
  SearchGrid();

  std::vector<unsigned int> accum_;
  under_type *vals_;

  // Returns a SearchGrid
  static Persistent *maker() { return new SearchGrid(); }
};


} // namespace SCIRun

#endif // SCI_project_SearchGrid_h
