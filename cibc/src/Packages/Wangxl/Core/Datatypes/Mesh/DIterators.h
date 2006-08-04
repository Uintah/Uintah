#ifndef SCI_Wangxl_Datatypes_Mesh_DIterators_h
#define SCI_Wangxl_Datatypes_Mesh_DIterators_h

#include <utility>
#include <iterator>

#include <Packages/Wangxl/Core/Datatypes/Mesh/DVertex.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DEdge.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DFacet.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DCell.h>

namespace Wangxl {

using namespace SCIRun;

/*template < class Gt, class Tds > class Triangulation_3;
template < class Gt, class Tds > class Triangulation_cell_3;
template < class Gt, class Tds > class Triangulation_vertex_3;
template < class Gt, class Tds > class Triangulation_3;
template < class Gt, class Tds > class Triangulation_cell_iterator_3;
template < class Gt, class Tds > class Triangulation_facet_iterator_3;
template < class Gt, class Tds > class Triangulation_edge_iterator_3;
template < class Gt, class Tds > class Triangulation_vertex_iterator_3;
*/
class Delaunay;

template <class Mesh>
class DCellIterator
{
public:
  typedef DCell       value_type;
  typedef DCell *     pointer;
  typedef DCell &     reference;
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef std::bidirectional_iterator_tag   iterator_category;

  /*  typedef typename Tds::Cell Ctds;
  typedef typename Tds::Cell_iterator Iterator_base;

  typedef typename Triangulation_3<Gt,Tds>::Cell Cell;
  typedef typename Triangulation_3<Gt,Tds>::Vertex Vertex;
  typedef typename Triangulation_3<Gt,Tds>::Vertex_handle Vertex_handle;  
  typedef typename Triangulation_3<Gt,Tds>::Cell_handle Cell_handle;
  typedef Triangulation_3<Gt,Tds> Triang;

  typedef DCellIterator<Gt,Tds> Cell_iterator;
  */
  DCellIterator() : d_it(), d_dln(0), d_inf(true) {}
        
  DCellIterator(const Delaunay* dln, bool inf)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh)),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(inf)
    { 
      if (! d_inf) {
	while ( // ( d_it != d_dln->_tds.cells_end() ) &&
	       // useless : there must be at least one finite cell
	       // since precond in d_it : dimension == 3
	       d_dln->is_infinite( (DCell *) &(*(d_it)) ) )
	  { ++d_it; }
      }
    }
  
  // for past-end iterator
  // does not need to find a finite cell
  DCellIterator(const Delaunay* dln)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh), 1),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(true) { }
       
  DCellIterator(const DCellIterator& cit)
    : d_it(cit.d_it), d_dln(cit.d_dln), d_inf(cit.d_inf) {}
        
  DCellIterator& operator=(const DCellIterator& cit)
  { 
    d_it = cit.d_it;
    d_dln = cit.d_dln;
    d_inf = cit.d_inf;
    return *this;
  }
  
  bool operator==(const DCellIterator& cit) const
  {
    if ( d_dln != cit.d_dln ) 
      return false;

    if ( ( d_it == d_dln->d_mesh.cells_end() ) 
	 || ( cit.d_it == d_dln->d_mesh.cells_end() ) ) 
      return ( d_it == cit.d_it );

    return ( ( d_it == cit.d_it ) && ( d_inf == cit.d_inf ) );
  }

  bool
  operator!=(const DCellIterator & cit)
  {
    return ( !(*this == cit) );
  }

  DCellIterator& operator++()
  {
    if (d_inf) {
      ++d_it;
    }
    else {
      do {
	++d_it; 
      } while ( ( d_it != d_dln->d_mesh.cells_end() )
		&& d_dln->is_infinite( (DCell *) &(*(d_it)) ) );
    }
    return *this;   
  }

  DCellIterator& operator--()
  {
    if (d_inf) {
      --d_it;
    }
    else{
      do {
	--d_it;
      } while ( ( d_it != d_dln->d_mesh.cells_end() )
		&& d_dln->is_infinite( (DCell *) &(*(d_it)) ) );
    }
    return *this;   
  }

  DCellIterator operator++(int)
  {
    DCellIterator tmp(*this);
    ++(*this);
    return tmp;
  }
        
  DCellIterator operator--(int)
  {
    DCellIterator tmp(*this);
    --(*this);
    return tmp;
  }
        
  DCell& operator*() const
  {
    return (DCell&)(*d_it);
  }

  DCell* operator->() const
  {
    return (DCell*)( &(*d_it) );
  }
     
private: 
  VMCellIterator<Mesh> d_it;
  Delaunay* d_dln;
  bool d_inf; // if d_inf == true, traverses all cells
               // else only traverses finite cells
};


template <class Mesh>
class DVertexIterator
{
public:
  typedef DVertex       value_type;
  typedef DVertex *     pointer;
  typedef DVertex &     reference;
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef std::bidirectional_iterator_tag   iterator_category;

  /*  typedef typename Tds::Vertex Ve;
  typedef typename Tds::Vertex_iterator Iterator_base;

  typedef typename Triangulation_3<Gt,Tds>::Vertex Vertex;
  typedef typename Triangulation_3<Gt,Tds>::Vertex_handle Vertex_handle;
  typedef Triangulation_3<Gt,Tds> Triang;

  typedef DVertexIterator<Gt,Tds> Vertex_iterator;
  */
  DVertexIterator() : d_it(), d_dln(0), d_inf(true) {}
        
  DVertexIterator(const Delaunay* dln, bool inf)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh)),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(inf)
    { 
      if (! d_inf) {
	if ( d_dln->is_infinite( (DVertex *) &(*(d_it)) ) ) {
	  ++d_it;
	}
      }
    }
        
  // for past-end iterator
  // does not need to find a finite cell
  DVertexIterator(const Delaunay* dln)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh), 1), d_dln(const_cast<Delaunay*>(dln))
  { }
       
  DVertexIterator(const DVertexIterator& vi)
    : d_it(vi.d_it), d_dln(vi.d_dln), d_inf(vi.d_inf) {}
        
  DVertexIterator& operator=(const DVertexIterator& vi)
  { 
    d_it = vi.d_it;
    d_dln = vi.d_dln;
    d_inf = vi.d_inf;
    return *this;
  }
  
  bool operator==(const DVertexIterator & vi) const
  {
    if ( d_dln != vi.d_dln ) 
      return false;

    if ( ( d_it == d_dln->d_mesh.vertices_end() ) 
	 || ( vi.d_it == d_dln->d_mesh.vertices_end() ) ) 
      return ( d_it == vi.d_it );

    return ( ( d_it == vi.d_it ) && ( d_inf == vi.d_inf ) );
  }

  bool operator!=(const DVertexIterator& vi)
  {
    return ( !(*this == vi) );
  }

  DVertexIterator& operator++()
  {
    if (d_inf) {
      ++d_it;
    }
    else {
      ++d_it; 
      if ( d_dln->is_infinite( (DVertex*) &(*(d_it)) ) ) {
	++d_it;
      }
    }
    return *this;   
  }

  DVertexIterator&  operator--()
  {
    if (d_inf) {
      --d_it;
    }
    else{
      --d_it;
      if ( d_dln->is_infinite( (DVertex*) &(*(d_it)) ) ) {
	--d_it;
      }
    }
    return *this;   
  }

  DVertexIterator operator++(int)
  {
    DVertexIterator tmp(*this);
    ++(*this);
    return tmp;
  }
        
  DVertexIterator operator--(int)
  {
    DVertexIterator tmp(*this);
    --(*this);
    return tmp;
  }

  DVertex& operator*() const
  {
    return (DVertex&)(*d_it);
  }

  DVertex* operator->() const
  {
    return   (DVertex*)( &(*d_it) );
  }
     
private:
  VMVertexIterator<Mesh> d_it;
  Delaunay* d_dln;
  bool d_inf; // if d_inf == true, traverses all vertices
               // else only traverses finite vertices
};


template <class Mesh>
class DEdgeIterator
{
public:
  typedef DEdge       value_type;
  typedef DEdge*      pointer;
  typedef DEdge&      reference;
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef std::bidirectional_iterator_tag   iterator_category;

  /*  typedef typename Tds::Edge Etds;
  typedef typename Tds::Edge_iterator  Iterator_base;

  typedef typename Triangulation_3<Gt,Tds>::Edge Edge;
  typedef typename Triangulation_3<Gt,Tds>::DCellHandle DCellHandle;
  typedef typename Triangulation_3<Gt,Tds>::Cell Cell;
  typedef Triangulation_3<Gt,Tds> Triang;

  typedef Triangulation_edge_iterator_3<Gt,Tds>      Edge_iterator;
  */
  DEdgeIterator() : d_it(), d_dln(0), d_inf(true) {}
        
  DEdgeIterator(const Delaunay* dln, bool inf)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh)),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(inf)
  { 
    if (! d_inf) {
      while ( // ( d_it != d_dln->d_mesh.cells_end() ) &&
	     // useless : there must be at least one finite cell
	     // since precond in d_it : dimension == 3
	     d_dln->is_infinite(make_triple( (DCell*)((*d_it).first), (*d_it).second, (*d_it).third ) ) )
      { ++d_it; }
    }
  }
        
  DEdgeIterator(const Delaunay* dln)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh), 1),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(true)
    // d_inf is initialized but should never be used
  { }
       
  DEdgeIterator(const DEdgeIterator& ei)
    : d_it(ei.d_it), d_dln(ei.d_dln), d_inf(ei.d_inf) {}
        
  DEdgeIterator& operator=(const DEdgeIterator& ei)
  { 
    d_it = ei.d_it;
    d_dln = ei.d_dln;
    d_inf = ei.d_inf;
    return *this;
  }
  
  bool operator==(const DEdgeIterator& ei) const
  {
    if ( d_dln != ei.d_dln ) 
      return false;

    if ( ( d_it == d_dln->d_mesh.edges_end() ) 
	 || ( ei.d_it == d_dln->d_mesh.edges_end() ) ) 
      return ( d_it == ei.d_it );

    return ( ( d_it == ei.d_it ) && ( d_inf == ei.d_inf ) );
  }

  bool operator!=(const DEdgeIterator& ei)
  {
    return !(*this == ei);
  }

  DEdgeIterator& operator++()
  {
    if (d_inf) {
      ++d_it;
    }
    else {
      do {
	++d_it; 
      } while ( ( d_it != d_dln->d_mesh.edges_end() )
		&& d_dln->is_infinite(make_triple( (DCell *) ((*d_it).first),
				       (*d_it).second, (*d_it).third ) ) );
    }
    return *this;   
  }

  DEdgeIterator&  operator--()
  {
    if (d_inf) {
      --d_it;
    }
    else{
      do {
	--d_it;
      } while ( ( d_it != d_dln->d_mesh.edges_end() )
	      && d_dln->is_infinite(make_triple( (DCell *) ((*d_it).first),
						 (*d_it).second,
						 (*d_it).third ) ) );
    }
    return *this;   
  }

  DEdgeIterator operator++(int)
  {
    DEdgeIterator tmp(*this);
    ++(*this);
    return tmp;
  }
        
  DEdgeIterator operator--(int)
  {
    DEdgeIterator tmp(*this);
    --(*this);
    return tmp;
  }
        
  DEdge operator*() const
  {
    //    DCellHandle ch = (DCell*)( (*d_it).first );
    return make_triple( (DCell*)((*d_it).first), (*d_it).second, (*d_it).third );
  }
     
private:
  VMEdgeIterator<Mesh> d_it ;
  Delaunay* d_dln;
  bool d_inf; // if d_inf == true, traverses all edges
               // else only traverses finite edges
};

template <class Mesh>
class DFacetIterator
{
public:
  typedef DFacet       value_type;
  typedef DFacet *     pointer;
  typedef DFacet &     reference;
  typedef size_t     size_type;
  typedef ptrdiff_t  difference_type;
  typedef std::bidirectional_iterator_tag   iterator_category;

  /*  typedef typename Tds::Facet Ftds;
  typedef typename Tds::Facet_iterator  Iterator_base;

  typedef typename Triangulation_3<Gt,Tds>::Facet Facet;
  typedef typename Triangulation_3<Gt,Tds>::DCellHandle DCellHandle;
  typedef typename Triangulation_3<Gt,Tds>::Cell Cell;
  typedef Triangulation_3<Gt,Tds> Triang;

  typedef Triangulation_facet_iterator_3<Gt,Tds> Facet_iterator;
  */
  DFacetIterator() : d_it(), d_dln(0), d_inf(true) {}
        
  DFacetIterator(const Delaunay* dln, bool inf)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh)),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(inf)
  {
    if (! d_inf) {
      while ( // ( d_it != d_dln->d_mesh.cells_end() ) &&
	     // useless : there must be at least one finite cell
	     // since precond in d_it : dimension == 3
	     d_dln->is_infinite(std::make_pair( (DCell*) ((*d_it).first),
						(*d_it).second ) ) )
	{ ++d_it; }
    }
  }
        
  DFacetIterator(const Delaunay* dln)
    : d_it( &(const_cast<Delaunay*>(dln)->d_mesh), 1),
      d_dln(const_cast<Delaunay*>(dln)), d_inf(true)
  // d_inf is initialized but should never be used
  { }
       
  DFacetIterator(const DFacetIterator& fi)
    : d_it(fi.d_it), d_dln(fi.d_dln), d_inf(fi.d_inf)
  {}
        
  DFacetIterator& operator=(const DFacetIterator& fi)
  { 
    d_it = fi.d_it;
    d_dln = fi.d_dln;
    d_inf = fi.d_inf;
    return *this;
  }
  
  bool operator==(const DFacetIterator& fi) const
  {
    if ( d_dln != fi.d_dln ) 
      return false;

    if ( ( d_it == d_dln->d_mesh.facets_end() ) 
	 || ( fi.d_it == d_dln->d_mesh.facets_end() ) ) 
      return ( d_it == fi.d_it );

    return ( ( d_it == fi.d_it ) && ( d_inf == fi.d_inf ) );
  }

  bool operator!=(const DFacetIterator& fi)
  {
    return !(*this == fi);
  }

  DFacetIterator& operator++()
  {
    if (d_inf) {
      ++d_it;
    }
    else {
      do {
	++d_it; 
      } while ( ( d_it != d_dln->d_mesh.facets_end() )
		&& d_dln->is_infinite(std::make_pair( (DCell*) ((*d_it).first),
						      (*d_it).second ) ) );
    }
    return *this;
  }

  DFacetIterator& operator--()
  {
    if (d_inf) {
      --d_it;
    }
    else{
      do {
	--d_it;
      } while ( ( d_it != d_dln->d_mesh.facets_end() )
		&& d_dln->is_infinite(std::make_pair( (DCell*) ((*d_it).first),
				      (*d_it).second ) ) );
    }
    return *this;   
  }

  DFacetIterator operator++(int)
  {
    DFacetIterator tmp(*this);
    ++(*this);
    return tmp;
  }
        
  DFacetIterator operator--(int)
  {
    DFacetIterator tmp(*this);
    --(*this);
    return tmp;
  }
        
  DFacet operator*() const
  {
    //    DCellHandle ch = (DCell*)( (*d_it).first );
    return std::make_pair( (DCell*)((*d_it).first), (*d_it).second );
  }
     
private:
  VMFacetIterator<Mesh> d_it ;
  Delaunay* d_dln;
  bool d_inf; // if d_inf == true, traverses all facets
               // else only traverses finite facets
};

}

#endif




