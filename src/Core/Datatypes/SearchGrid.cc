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
 *  SearchGrid.cc: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan &&
 *   Chris Moulding
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#include <Core/Datatypes/SearchGrid.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MusilRNG.h>
#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;


SearchGridBase::SearchGridBase(unsigned x, unsigned y, unsigned z,
			       const Point &min, const Point &max)
  : ni_(x), nj_(y), nk_(z)
{
  transform_.pre_scale(Vector(1.0 / x, 1.0 / y, 1.0 / z));
  transform_.pre_scale(max - min);

  transform_.pre_translate(min.asVector());
  transform_.compute_imat();
}

SearchGridBase::SearchGridBase(unsigned x, unsigned y, unsigned z,
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
SearchGridBase::locate(unsigned int &i, unsigned int &j, unsigned int &k,
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

  i = (unsigned int)rx;
  j = (unsigned int)ry;
  k = (unsigned int)rz;
  return true;
}


void
SearchGridBase::unsafe_locate(unsigned int &i, unsigned int &j,
			      unsigned int &k, const Point &p) const
{
  Point r;
  transform_.unproject(p, r);
  
  r.x(floor(r.x()));
  r.y(floor(r.y()));
  r.z(floor(r.z()));

  i = (unsigned int)r.x();
  j = (unsigned int)r.y();
  k = (unsigned int)r.z();
}


SearchGridConstructor::SearchGridConstructor(unsigned int x,
					     unsigned int y,
					     unsigned int z,
					     const Point &min,
					     const Point &max)
  : SearchGridBase(x, y, z, min, max), size_(0)
{
  bin_.resize(x * y * z);
}


void
SearchGridConstructor::insert(under_type val, const BBox &bbox)
{
  unsigned int mini, minj, mink, maxi, maxj, maxk;

  unsafe_locate(mini, minj, mink, bbox.min());
  unsafe_locate(maxi, maxj, maxk, bbox.max());

  for (unsigned int i = mini; i <= maxi; i++)
  {
    for (unsigned int j = minj; j <= maxj; j++)
    {
      for (unsigned int k = mink; k <= maxk; k++)
      {
	bin_[linearize(i, j, k)].push_back(val);
	size_++;
      }
    }
  }
}


PersistentTypeID SearchGrid::type_id("SearchGrid", "Datatype", maker);


SearchGrid::SearchGrid()
  : SearchGridBase(1, 1, 1, Point(0.0, 0.0, 0.0), Point(1.0, 1.0, 1.0)),
    vals_(0)
{
}


SearchGrid::SearchGrid(const SearchGridConstructor &c)
  : SearchGridBase(c.ni_, c.nj_, c.nk_, c.transform_)
{
  accum_.resize(ni_ * nj_ * nk_ + 1);
  vals_ = new under_type[c.size_];

  unsigned int counter = 0;
  accum_[0] = 0;
  for (unsigned int i = 0; i < ni_; i++)
  {
    for (unsigned int j = 0; j < nj_; j++)
    {
      for (unsigned int k = 0; k < nk_; k++)
      {
	// TODO:  Sort by size so more likely to get hit is checked first.
	list<under_type>::const_iterator itr = c.bin_[counter].begin();
	unsigned int size = 0;
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



SearchGrid::~SearchGrid()
{
  if (vals_) { delete vals_; } vals_ = 0;
}


bool
SearchGrid::lookup(under_type **begin, under_type **end, const Point &p) const
{
  unsigned int i, j, k;
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


