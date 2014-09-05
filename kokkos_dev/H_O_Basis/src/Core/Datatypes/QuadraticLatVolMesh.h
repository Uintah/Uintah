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

//    File   : QuadraticLatVolMesh.h
//    Author : Michael Callahan
//    Date   : Sun Feb 24 14:25:39 2002

#ifndef Datatypes_QuadraticLatVolMesh_h
#define Datatypes_QuadraticLatVolMesh_h

#include <Core/Datatypes/LatVolMesh.h>

namespace SCIRun {

//! Base type for index types.
template <class N, class E>
struct SequentialIndex {
  typedef N node_type;
  typedef E edge_type;
  
  SequentialIndex(N n) :
    which_(0),
    node_index_(n)
  {}

  SequentialIndex(N n, E e) :
    which_(1),
    node_index_(n),
    edge_index_(e)
  {}

  //! Required interface for an Index.
  operator unsigned int() const
  {
    if (which_ == 0) 
    {
      return ((unsigned int)node_index_);
    }
    else
    {
      return ((unsigned int)node_index_) + ((unsigned int)edge_index_);
    }
  }

  unsigned char which_;
  N node_index_;
  E edge_index_;
};



template <class N, class E>
const TypeDescription* get_type_description(SequentialIndex<N, E>*)
{
  static TypeDescription* td = 0;

  if(!td){
    const TypeDescription *nsub = SCIRun::get_type_description((N*)0);
    const TypeDescription *esub = SCIRun::get_type_description((E*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(2);
    (*subs)[0] = nsub;
    (*subs)[1] = esub;
    td = scinew TypeDescription("SequentialIndex", subs, __FILE__, "SCIRun");
  }
  return td;
}

#define SEQUENTIALINDEX_VERSION 1

template<class N, class E>
void Pio(Piostream& stream, SequentialIndex<N, E>& data)
{
  Pio(stream, which_);
  Pio(stream, node_index_);
  Pio(stream, edge_index_);
}

template <class N, class E>
const string find_type_name(SequentialIndex<N, E> *)
{
  static const string name =
    string("SequentialIndex") + FTNS + find_type_name((N *)0) +
    FTNM + find_type_name((E *)0) + FTNE;
  return name;
}


template <class N, class E>
struct SequentialSize
{
  SequentialSize() {}
  SequentialSize(N nsize, E esize) : nsize_(nsize), esize_(esize) {}
  operator unsigned() const { return ((unsigned)nsize_) + ((unsigned)esize_); }

  N nsize_;
  E esize_;
};


template <class N, class E, class Index>
struct SequentialIter
{
  SequentialIter(N nbegin, N nend, E ebegin) :
    which_(0),
    niter_(nbegin),
    nend_(nend),
    eiter_(ebegin)
  {}

  const Index &operator *() const
  {
    if (which_ == 0)
    {
      return *niter_;
    }
    else
    {
      return *eiter_;
    }
  }

  bool operator ==(const SequentialIter<N, E, Index> &a) const
  {
    return niter_ == a.niter_ && eiter_ == a.eiter_;
  }

  bool operator !=(const SequentialIter<N, E, Index> &a) const
  {
    return !(*this == a);
  }

  SequentialIter<N, E, Index> &operator++()
  {
    if (which_ == 0)
    {
      if (niter_ == nend_)
      {
	which_ = 1;
	++eiter_;
      }
      else
      {
	++niter_;
      }
    }
    else
    {
      ++eiter_;
    }
    return *this;
  }

  unsigned char which_;
  N niter_;
  N nend_;
  E eiter_;

private:

  SequentialIter<N, E, Index> &operator++(int)
  {
    SequentialIter<N, E, Index> copy(*this);
    operator++();
    return copy;
  }
};



class SCICORESHARE QuadraticLatVolMesh : public LatVolMesh
{
public:

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef SequentialIndex<NodeIndex, EdgeIndex<unsigned int> > index_type;
    typedef SequentialIter<NodeIter, EdgeIterator<unsigned int>, index_type> iterator;
    typedef SequentialSize<NodeSize, EdgeIndex<unsigned int> >   size_type;
    typedef vector<index_type> array_type;
  };			

  QuadraticLatVolMesh();
  QuadraticLatVolMesh(const LatVolMesh &tv);
  QuadraticLatVolMesh(const QuadraticLatVolMesh &copy);

  virtual QuadraticLatVolMesh *clone() 
  { return new QuadraticLatVolMesh(*this); }
  virtual ~QuadraticLatVolMesh();

  void begin(Node::iterator &) const;
  void begin(Edge::iterator &) const;
  void begin(Face::iterator &) const;
  void begin(Cell::iterator &) const;

  void end(Node::iterator &) const;
  void end(Edge::iterator &) const;
  void end(Face::iterator &) const;
  void end(Cell::iterator &) const;

  void size(Node::size_type &) const;
  void size(Edge::size_type &) const;
  void size(Face::size_type &) const;
  void size(Cell::size_type &) const;

  void to_index(Node::index_type &index, unsigned int i);
  void to_index(Edge::index_type &index, unsigned int i) { index = i; }
  void to_index(Face::index_type &index, unsigned int i) { index = i; }
  void to_index(Cell::index_type &index, unsigned int i);

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, const Cell::index_type &idx) const;


  void get_center(Point &result, const Node::index_type &index) const;

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double *)
  { ASSERTFAIL("QuadraticLatVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , Face::array_type & , double *)
  { ASSERTFAIL("QuadraticLatVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, Cell::array_type &l, double *w)
  { return LatVolMesh::get_weights(p, l, w); }

#if 0
  //! get gradient relative to point p
  void get_gradient_basis(Cell::index_type ci, const Point& p,
			    Vector& g0, Vector& g1, Vector& g2, Vector& g3, 
			    Vector& g4, Vector& g5, Vector& g6, Vector& g7, 
			    Vector& g8, Vector& g9) const;

  //! gradient for gauss pts 
  double get_gradient_basis(Cell::index_type ci, int gaussPt, const Point&, 
			    Vector& g0, Vector& g1, Vector& g2, Vector& g3, 
			    Vector& g4, Vector& g5, Vector& g6, Vector& g7, 
			    Vector& g8, Vector& g9) const;
#endif

  void add_node_neighbors(Node::array_type &array, Node::index_type node, 
			  const vector<bool> &bc, bool apBC=true);


  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

private:


};

// Handle type for LatVolMesh mesh.
typedef LockingHandle<QuadraticLatVolMesh> QuadraticLatVolMeshHandle;
const TypeDescription* get_type_description(QuadraticLatVolMesh *);
const TypeDescription* get_type_description(QuadraticLatVolMesh::Node *);

} // namespace SCIRun


#endif // Datatypes_QuadraticLatVolMesh_h
