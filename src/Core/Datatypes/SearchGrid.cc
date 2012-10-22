/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
 *  SearchGrid.cc: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan &&
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *
 */

#include <Core/Datatypes/SearchGrid.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Math/MiscMath.h>
#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;


SearchGridBase::~SearchGridBase()
{
}


SearchGridBase::SearchGridBase(int x, int y, int z,
                               const Point &min, const Point &max)
  : ni_(x), nj_(y), nk_(z)
{
  transform_.pre_scale(Vector(1.0 / x, 1.0 / y, 1.0 / z));
  transform_.pre_scale(max - min);

  transform_.pre_translate(min.asVector());
  transform_.compute_imat();
}

SearchGridBase::SearchGridBase(int x, int y, int z,
                               const Transform &t)
  : ni_(x), nj_(y), nk_(z), transform_(t)
{
}


void
SearchGridBase::transform(const Transform &t)
{
  transform_.pre_trans(t);
}

void
SearchGridBase::get_canonical_transform(Transform &t) 
{
  t = transform_;
  t.post_scale(Vector(ni_, nj_, nk_));
}


bool
SearchGridBase::locate(int &i, int &j, int &k,
                       const Point &p) const
{
  const Point r = transform_.unproject(p);
  
  const double rx = floor(r.x());
  const double ry = floor(r.y());
  const double rz = floor(r.z());

  // Clamp in double space to avoid overflow errors.
  if (rx < 0.0      || ry < 0.0      || rz < 0.0    ||
      rx >= ni_     || ry >= nj_     || rz >= nk_   )
  {
    return false;
  }

  i = (int)rx;
  j = (int)ry;
  k = (int)rz;
  return true;
}


void
SearchGridBase::unsafe_locate(int &i, int &j,
                              int &k, const Point &p) const
{
  Point r;
  transform_.unproject(p, r);
  
  i = (int)r.x();
  j = (int)r.y();
  k = (int)r.z();
}


SearchGridConstructor::SearchGridConstructor(int x,
                                             int y,
                                             int z,
                                             const Point &min,
                                             const Point &max)
  : SearchGridBase(x, y, z, min, max), size_(0)
{
  bin_.resize(x * y * z);
}


SearchGridConstructor::~SearchGridConstructor()
{
}


void
SearchGridConstructor::insert(under_type val, const BBox &bbox)
{
  int mini, minj, mink, maxi, maxj, maxk;

  unsafe_locate(mini, minj, mink, bbox.min());
  unsafe_locate(maxi, maxj, maxk, bbox.max());

  for (int i = mini; i <= maxi; i++)
  {
    for (int j = minj; j <= maxj; j++)
    {
      for (int k = mink; k <= maxk; k++)
      {
        bin_[linearize(i, j, k)].push_back(val);
        size_++;
      }
    }
  }
}


void
SearchGridConstructor::remove(under_type val, const BBox &bbox)
{
  int mini, minj, mink, maxi, maxj, maxk;

  unsafe_locate(mini, minj, mink, bbox.min());
  unsafe_locate(maxi, maxj, maxk, bbox.max());

  for (int i = mini; i <= maxi; i++)
  {
    for (int j = minj; j <= maxj; j++)
    {
      for (int k = mink; k <= maxk; k++)
      {
        bin_[linearize(i, j, k)].remove(val);
        size_++;
      }
    }
  }
}


bool
SearchGridConstructor::lookup(const list<under_type> *&candidates,
                              const Point &p) const
{
  int i, j, k;
  if (locate(i, j, k, p))
  {
    candidates = &(bin_[linearize(i, j, k)]);
    return true;
  }
  return false;
}


void
SearchGridConstructor::lookup_ijk(const list<under_type> *&candidates,
                                  int i, int j, int k) const
{
  candidates = &(bin_[linearize(i, j, k)]);
}


double
SearchGridConstructor::min_distance_squared(const Point &p,
                                            int i, int j, int k) const
{
  Point r;
  transform_.unproject(p, r);

  // Splat the point onto the cell.
  if (r.x() < i) { r.x(i); }
  else if (r.x() > i+1) { r.x(i+1); }

  if (r.y() < j) { r.y(j); }
  else if (r.y() > j+1) { r.y(j+1); }

  if (r.z() < k) { r.z(k); }
  else if (r.z() > k+1) { r.z(k+1); }
  
  // Project the cell intersection back to world space.
  Point q;
  transform_.project(r, q);
  
  // Return distance from point to projected cell point.
  return (p - q).length2();
}


PersistentTypeID SearchGrid::type_id("SearchGrid", "Datatype", maker);


SearchGrid::SearchGrid()
  : SearchGridBase(1, 1, 1, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
    vals_(0),
    vals_size_(0)
{
}


SearchGrid::SearchGrid(const SearchGridConstructor &c)
  : SearchGridBase(c.ni_, c.nj_, c.nk_, c.transform_)
{
  accum_.resize(ni_ * nj_ * nk_ + 1);
  vals_size_ = c.size_;
  vals_ = new under_type[vals_size_];
  

  int counter = 0;
  accum_[0] = 0;
  for (int i = 0; i < ni_; i++)
  {
    for (int j = 0; j < nj_; j++)
    {
      for (int k = 0; k < nk_; k++)
      {
        // NOTE: Sort by size so more likely to get hit is checked first?
        // NOTE: Quick testing showed a 3% performance gain in heavy
        // search/build ratio test.  Build time goes up.  Also makes
        // edge tests less consistent.  We currently pick lowest
        // element index on an between-element hit because of the way
        // these things are built.
        list<under_type>::const_iterator itr = c.bin_[counter].begin();
        int size = 0;
        while (itr != c.bin_[counter].end())
        {
          vals_[accum_[counter] + size] = *itr;
          size++;
          ++itr;
        }
        accum_[counter+1] = accum_[counter] + size;
        counter++;
      }
    }
  }
}


SearchGrid::SearchGrid(const SearchGrid &c)
  : SearchGridBase(c.ni_, c.nj_, c.nk_, c.transform_),
    accum_(c.accum_),
    vals_size_(c.vals_size_)
{
  vals_ = new under_type[vals_size_];
  for (unsigned int i = 0; i < vals_size_; i++)
  {
    vals_[i] = c.vals_[i];
  }
}


SearchGrid::~SearchGrid()
{
  if (vals_) { delete vals_; } vals_ = 0; vals_size_ = 0;
}


bool
SearchGrid::lookup(under_type **begin, under_type **end, const Point &p) const
{
  int i, j, k;
  if (locate(i, j, k, p))
  {
    const unsigned int index = linearize(i, j, k);
    *begin = vals_ + accum_[index];
    *end = vals_ + accum_[index+1];
    return true;
  }
  return false;
}


#define SEARCHGRID_VERSION 1

void
SearchGrid::io(Piostream& stream)
{
  stream.begin_class("SearchGrid", SEARCHGRID_VERSION);

  // IO data members, in order
  Pio(stream, ni_);
  Pio(stream, nj_);
  Pio(stream, nk_);

  Pio(stream, transform_);

  //Pio(stream, accum_);
  //Pio(stream, vals_);

  stream.end_class();
}


} // namespace SCIRun


