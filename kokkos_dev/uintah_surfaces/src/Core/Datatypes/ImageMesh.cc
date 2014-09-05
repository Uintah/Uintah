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
 *  ImageMesh.cc: Templated Mesh defined on a 3D Regular Grid
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

#include <Core/Datatypes/ImageMesh.h>
#include <Core/Geometry/BBox.h>
#include <Core/Math/MusilRNG.h>
#include <vector>

namespace SCIRun {

using namespace std;


PersistentTypeID ImageMesh::type_id("ImageMesh", "Mesh", maker);


ImageMesh::ImageMesh(unsigned i, unsigned j,
		     const Point &min, const Point &max)
  : min_i_(0), min_j_(0), ni_(i), nj_(j)
{
  transform_.pre_scale(Vector(1.0 / (i-1.0), 1.0 / (j-1.0), 1.0));
  transform_.pre_scale(max - min);
  transform_.pre_translate(Vector(min));
  transform_.compute_imat();

  normal_ = Vector(0.0, 0.0, 0.0);
  transform_.project_normal(normal_);
  normal_.safe_normalize();
}


BBox
ImageMesh::get_bounding_box() const
{
  Point p0(0.0,   0.0,   0.0);
  Point p1(ni_-1, 0.0,   0.0);
  Point p2(ni_-1, nj_-1, 0.0);
  Point p3(0.0,   nj_-1, 0.0);
  
  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  result.extend(transform_.project(p2));
  result.extend(transform_.project(p3));
  return result;
}

Vector ImageMesh::diagonal() const
{
  return get_bounding_box().diagonal();
}

void
ImageMesh::get_canonical_transform(Transform &t) 
{
  t = transform_;
  t.post_scale(Vector(ni_ - 1.0, nj_ - 1.0, 1.0));
}

bool
ImageMesh::synchronize(unsigned int flag)
{
  if (flag & NORMALS_E)
  {
    normal_ = Vector(0.0, 0.0, 0.0);
    transform_.project_normal(normal_);
    normal_.safe_normalize();
    return true;
  }
  return false;
}

void
ImageMesh::transform(const Transform &t)
{
  transform_.pre_trans(t);
}

bool
ImageMesh::get_min(vector<unsigned int> &array) const
{
  array.resize(2);
  array.clear();

  array.push_back(min_i_);
  array.push_back(min_j_);

  return true;
}

bool
ImageMesh::get_dim(vector<unsigned int> &array) const
{
  array.resize(2);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);

  return true;
}

void
ImageMesh::set_min(vector<unsigned int> min)
{
  min_i_ = min[0];
  min_j_ = min[1];
}

void
ImageMesh::set_dim(vector<unsigned int> dim)
{
  ni_ = dim[0];
  nj_ = dim[1];
}

void
ImageMesh::get_nodes(Node::array_type &array, Face::index_type idx) const
{
  array.resize(4);

  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1;

  array[0].mesh_ = idx.mesh_;
  array[1].mesh_ = idx.mesh_;
  array[2].mesh_ = idx.mesh_;
  array[3].mesh_ = idx.mesh_;
}


void
ImageMesh::get_nodes(Node::array_type &array, Edge::index_type idx) const
{
  array.resize(2);

  const int j_idx = idx - (ni_-1) * nj_;
  if (j_idx >= 0)
  {
    const int i = j_idx / (nj_ - 1);
    const int j = j_idx % (nj_ - 1);
    array[0] = Node::index_type(this, i, j);
    array[1] = Node::index_type(this, i, j+1);
  }
  else
  {
    const int i = idx % (ni_ - 1);
    const int j = idx / (ni_ - 1);
    array[0] = Node::index_type(this, i, j);
    array[1] = Node::index_type(this, i+1, j);
  }
}


void
ImageMesh::get_edges(Edge::array_type &array, Face::index_type idx) const
{
  array.resize(4);  

  const int j_idx = (ni_-1) * nj_;

  array[0] = idx.i_             + idx.j_    *(ni_-1); 
  array[1] = idx.i_             + (idx.j_+1)*(ni_-1);
  array[2] = idx.i_    *(nj_-1) + idx.j_+ j_idx;
  array[3] = (idx.i_+1)*(nj_-1) + idx.j_+ j_idx;
}
  


bool
ImageMesh::get_neighbor(Face::index_type &neighbor,
			Face::index_type from,
			Edge::index_type edge) const
{
  neighbor.mesh_ = this;
  const int j_idx = edge - (ni_-1) * nj_;
  if (j_idx >= 0)
  {
    const unsigned int i = j_idx / (nj_ - 1);
    if (i == 0 || i == ni_-1) 
      return false;
    neighbor.j_ = from.j_;
    if (i == from.i_)
      neighbor.i_ = from.i_ - 1;
    else
      neighbor.i_ = from.i_ + 1;
  }
  else
  {
    const unsigned int j = edge / (ni_ - 1);;
    if (j == 0 || j == nj_-1) 
      return false;
    neighbor.i_ = from.i_;
    if (j == from.j_)
      neighbor.j_ = from.j_ - 1;
    else
      neighbor.j_ = from.j_ + 1; 
  }
  return true;
}

void 
ImageMesh::get_neighbors(Face::array_type &array, Face::index_type idx) const
{
  Edge::array_type edges;
  get_edges(edges, idx);
  array.clear();
  Edge::array_type::iterator iter = edges.begin();
  while(iter != edges.end()) {
    Face::index_type nbor;
    if (get_neighbor(nbor, idx, *iter)) {
      array.push_back(nbor);
    }
    ++iter;
  }
}

int
ImageMesh::get_valence(Edge::index_type idx) const
{
  const int j_idx = idx - (ni_-1) * nj_;
  if (j_idx >= 0)
  {
    const unsigned int i = j_idx / (nj_ - 1);
    return (i == 0 || i == ni_ - 1) ? 1 : 2;
  }
  else
  {
    const unsigned int j = idx / (ni_ - 1);
    return (j == 0 || j == nj_ - 1) ? 1 : 2;
  }
}


//! return all face_indecies that overlap the BBox in arr.
void
ImageMesh::get_faces(Face::array_type &arr, const BBox &bbox)
{
  arr.clear();
  Face::index_type min;
  locate(min, bbox.min());
  Face::index_type max;
  locate(max, bbox.max());

  if (max.i_ >= ni_ - 1) max.i_ = ni_ - 2;
  if (max.j_ >= nj_ - 1) max.j_ = nj_ - 2;

  for (unsigned i = min.i_; i <= max.i_; i++) {
    for (unsigned j = min.j_; j <= max.j_; j++) {
      arr.push_back(Face::index_type(this, i,j));
    }
  }
}


void
ImageMesh::get_center(Point &result, const Node::index_type &idx) const
{
  Point p(idx.i_, idx.j_, 0.0);
  result = transform_.project(p);
}


void
ImageMesh::get_center(Point &result, Edge::index_type idx) const
{
  Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);
  
  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


void
ImageMesh::get_center(Point &result, const Face::index_type &idx) const
{
  Point p(idx.i_ + 0.5, idx.j_ + 0.5, 0.0);
  result = transform_.project(p);
}


bool
ImageMesh::locate(Face::index_type &face, const Point &p)
{
  const Point r = transform_.unproject(p);

  const double rx = floor(r.x());
  const double ry = floor(r.y());

  // Clip before int conversion to avoid overflow errors.
  if (rx < 0.0 || ry < 0.0 || rx >= (ni_-1) || ry >= (nj_-1))
  {
    return false;
  }

  face.i_ = (unsigned int)rx;
  face.j_ = (unsigned int)ry;
  face.mesh_ = this;

  return true;
}


bool
ImageMesh::locate(Node::index_type &node, const Point &p)
{
  const Point r = transform_.unproject(p);

  const double rx = floor(r.x() + 0.5);
  const double ry = floor(r.y() + 0.5);

  // Clip before int conversion to avoid overflow errors.
  if (rx < 0.0 || ry < 0.0 || rx >= ni_ || ry >= nj_)
  {
    return false;
  }

  node.i_ = (unsigned int)rx;
  node.j_ = (unsigned int)ry;
  node.mesh_ = this;

  return true;
}


int
ImageMesh::get_weights(const Point &p, Node::array_type &locs, double *w)
{
  const Point r = transform_.unproject(p);
  double ii = r.x();
  double jj = r.y();
  if (ii>(ni_-1) && (ii-(1.e-10))<(ni_-1)) ii=ni_-1-(1.e-10);
  if (jj>(nj_-1) && (jj-(1.e-10))<(nj_-1)) jj=nj_-1-(1.e-10);
  if (ii<0 && ii>(-1.e-10)) ii=0;
  if (jj<0 && jj>(-1.e-10)) jj=0;

  Node::index_type node0;
  node0.i_ = (unsigned int)floor(ii);
  node0.j_ = (unsigned int)floor(jj);
  node0.mesh_ = this;

  if (node0.i_ < (ni_-1) && node0.i_ >= 0 &&
      node0.j_ < (nj_-1) && node0.j_ >= 0)
  {
    const double dx1 = ii - node0.i_;
    const double dy1 = jj - node0.j_;
    const double dx0 = 1.0 - dx1;
    const double dy0 = 1.0 - dy1;

    locs.resize(4);
    locs[0] = node0;
    
    locs[1].i_ = node0.i_ + 1;
    locs[1].j_ = node0.j_ + 0;
    locs[1].mesh_ = this;

    locs[2].i_ = node0.i_ + 1;
    locs[2].j_ = node0.j_ + 1;
    locs[2].mesh_ = this;
    
    locs[3].i_ = node0.i_ + 0;
    locs[3].j_ = node0.j_ + 1;
    locs[3].mesh_ = this;
    
    w[0] = dx0 * dy0;
    w[1] = dx1 * dy0;
    w[2] = dx1 * dy1;
    w[3] = dx0 * dy1;
    return 4;
  }
  return 0;
}


int
ImageMesh::get_weights(const Point &p, Face::array_type &l, double *w)
{
  Face::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


/* To generate a random point inside of a triangle, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */
void ImageMesh::get_random_point(Point &p, const Face::index_type &ci,
				   int seed) const
{
  static MusilRNG rng;

  // get the positions of the vertices
  Node::array_type ra;
  get_nodes(ra,ci);
  Point p00, p10, p11, p01;
  get_center(p00,ra[0]);
  get_center(p10,ra[1]);
  get_center(p11,ra[2]);
  get_center(p01,ra[3]);
  Vector dx=p10-p00;
  Vector dy=p01-p00;
  // generate the barrycentric coordinates
  double u,v;
  if (seed) {
    MusilRNG rng1(seed);
    rng1();
    u = rng1(); 
    v = rng1();
  } else {
    u = rng(); 
    v = rng();
  }

  // compute the position of the random point
  p = p00+dx*u+dy*v;
}


const TypeDescription* get_type_description(ImageMeshINodeIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("ImageMesh::INodeIndex",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription* get_type_description(ImageMeshIFaceIndex *)
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription("ImageMesh::IFaceIndex",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


void
Pio(Piostream& stream, ImageMeshINodeIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    stream.end_cheap_delim();
}

void
Pio(Piostream& stream, ImageMeshIFaceIndex& n)
{
    stream.begin_cheap_delim();
    Pio(stream, n.i_);
    Pio(stream, n.j_);
    stream.end_cheap_delim();
}



const string find_type_name(ImageMeshINodeIndex *)
{
  static string name = "ImageMesh::INodeIndex";
  return name;
}
const string find_type_name(ImageMeshIFaceIndex *)
{
  static string name = "ImageMesh::IFaceIndex";
  return name;
}


#define IMAGEMESH_VERSION 1

void
ImageMesh::io(Piostream& stream)
{
  stream.begin_class(type_name(-1), IMAGEMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  Pio(stream, nj_);

  stream.end_class();
}

const string
ImageMesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "ImageMesh";
  return name;
}


void
ImageMesh::begin(ImageMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_i_, min_j_);
}

void
ImageMesh::end(ImageMesh::Node::iterator &itr) const
{
  itr = Node::iterator(this, min_i_, min_j_ + nj_);
}

void
ImageMesh::size(ImageMesh::Node::size_type &s) const
{
  s = Node::size_type(ni_, nj_);
}


void
ImageMesh::to_index(ImageMesh::Node::index_type &idx, unsigned int a)
{
  const unsigned int i = a % ni_;
  const unsigned int j = a / ni_;
  idx = Node::index_type(this, i, j);

}


void
ImageMesh::begin(ImageMesh::Edge::iterator &itr) const
{
  itr = 0;
}

void
ImageMesh::end(ImageMesh::Edge::iterator &itr) const
{
  itr = (ni_-1) * (nj_) + (ni_) * (nj_ -1);
}

void
ImageMesh::size(ImageMesh::Edge::size_type &s) const
{
  s = (ni_-1) * (nj_) + (ni_) * (nj_ -1);
}

void
ImageMesh::begin(ImageMesh::Face::iterator &itr) const
{
  itr = Face::iterator(this,  min_i_, min_j_);
}

void
ImageMesh::end(ImageMesh::Face::iterator &itr) const
{
  itr = Face::iterator(this, min_i_, min_j_ + nj_ - 1);
}

void
ImageMesh::size(ImageMesh::Face::size_type &s) const
{
  s = Face::size_type(ni_-1, nj_-1);
}


void
ImageMesh::to_index(ImageMesh::Face::index_type &idx, unsigned int a)
{
  const unsigned int i = a % (ni_-1);
  const unsigned int j = a / (ni_-1);
  idx = Face::index_type(this, i, j);

}


void
ImageMesh::begin(ImageMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(0);
}

void
ImageMesh::end(ImageMesh::Cell::iterator &itr) const
{
  itr = Cell::iterator(0);
}

void
ImageMesh::size(ImageMesh::Cell::size_type &s) const
{
  s = Cell::size_type(0);
}


std::ostream& 
operator<<(std::ostream& os, const ImageMeshImageIndex& n) {
  os << "[" << n.i_ << "," << n.j_ << "]";
  return os;
}

std::ostream& 
operator<<(std::ostream& os, const ImageMeshImageSize& s) {
  os << (int)s << " (" << s.i_ << " x " << s.j_ << ")";
  return os;
}


const TypeDescription*
ImageMesh::get_type_description() const
{
  return SCIRun::get_type_description((ImageMesh *)0);
}

const TypeDescription*
get_type_description(ImageMesh::Node *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Node",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh::Edge *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Edge",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh::Face *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Face",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}

const TypeDescription*
get_type_description(ImageMesh::Cell *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    td = scinew TypeDescription("ImageMesh::Cell",
				TypeDescription::cc_to_h(__FILE__),
				"SCIRun");
  }
  return td;
}


ImageMeshImageIndex::operator unsigned() const
{ 
  ASSERT(mesh_);
  return i_ + j_*mesh_->ni_;
}

ImageMeshIFaceIndex::operator unsigned() const
{ 
  ASSERT(mesh_);
  return i_ + j_ * (mesh_->ni_-1);
}


ImageMeshINodeIter &
ImageMeshINodeIter::operator++()
{
  i_++;
  if (i_ >= mesh_->min_i_ + mesh_->ni_) {
    i_ = mesh_->min_i_;
    j_++;
  }
  return *this;
}


ImageMeshIFaceIter &
ImageMeshIFaceIter::operator++()
{
  i_++;
  if (i_ >= mesh_->min_i_+mesh_->ni_-1) {
    i_ = mesh_->min_i_;
    j_++;
  }
  return *this;
}


} // namespace SCIRun
