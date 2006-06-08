#ifndef SCI_Wangxl_Datatypes_Mesh_VMCirculators_h
#define SCI_Wangxl_Datatypes_Mesh_VMCirculators_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCell.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMFacet.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMEdge.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Utilities.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/CirculatorBase.h>

namespace Wangxl {

using namespace SCIRun;

template<class Mesh>
class VMCellCirculator
  : public Bidirectional_circulator_base<VMCell, ptrdiff_t, size_t>,
    public Utilities
{
  // circulates on cells around a given edge
public:
  /*
  typedef Tds_                  Tds;
  typedef typename Tds::Cell    Cell;
  typedef typename Tds::Edge    Edge;
  typedef typename Tds::Vertex  Vertex;

  typedef Triangulation_ds_cell_circulator_3<Tds> Cell_circulator;
  */
  VMCellCirculator() : d_mesh(0), d_c(0), d_s(0), d_t(0), pos(0) {}

   VMCellCirculator(const Mesh * mesh, VMCell* c, int s, int t)
    : d_mesh(const_cast<Mesh *>(mesh)), d_c(c), d_s(s), d_t(t)
  {
    assert( c != 0 && s >= 0 && s < 4 && t >= 0 && t < 4 );
    //     if ( _tds->dimension() <3 ) useless since precondition in tds
    //     incident_cells
    pos = c;
  }

   VMCellCirculator(const Mesh * mesh, const VMEdge & e)
    :  d_mesh(const_cast<Mesh *>(mesh)), d_c(e.first), d_s(e.second), d_t(e.third)
  {
    assert( e.first != 0 && e.second >=0 && e.second < 4 && e.third  >=0 && e.third  < 4);
    //     if ( _tds->dimension() <3 ) useless since precondition in tds
    //     incident_cells
      pos = e.first;
  }

  VMCellCirculator(const Mesh * mesh, VMCell* c, int s, int t, VMCell* start)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(c), d_s(s), d_t(t)
  {
    assert( c != 0 && s >= 0 && s < 4 && t >= 0 && t < 
	    4 && start->has_vertex( c->vertex(s) ) &&
	    start->has_vertex( c->vertex(t) ) );
    pos = start;
  }

   VMCellCirculator(const Mesh* mesh, const VMEdge & e, VMCell* start)
    : d_mesh(const_cast<Mesh *>(mesh)), d_c(e.first), d_s(e.second), d_t(e.third)
  {
    assert( e.first != 0 && e.second >=0 && e.second < 4 && e.third  >=0
	    && e.third  < 4 && start->has_vertex( e.first->vertex(e.second) ) &&
	    start->has_vertex( e.first->vertex(e.third) ) );
    pos = start;
  }

  VMCellCirculator(const VMCellCirculator & ccir)
    : d_mesh(ccir.d_mesh), d_c(ccir.d_c), d_s(ccir.d_s), d_t(ccir.d_t), pos(ccir.pos)
  {}

  VMCellCirculator & operator++()
  {
    assert( pos != NULL );
    //then dimension() cannot be < 3

    pos = pos->neighbor( next_around_edge( pos->index(d_c->vertex(d_s)),
					   pos->index(d_c->vertex(d_t)) ) );
    return *this;
  }

  VMCellCirculator operator++(int)
  {
    assert( pos != 0 );
    VMCellCirculator tmp(*this);
    ++(*this);
    return tmp;
  }

  VMCellCirculator & operator--()
  {
    assert( pos != 0 );

    pos = pos->neighbor( next_around_edge( pos->index(d_c->vertex(d_t)),
					   pos->index(d_c->vertex(d_s)) ) );
    return *this;
  }

  VMCellCirculator operator--(int)
  {
    assert( pos != 0 );
    VMCellCirculator tmp(*this);
    --(*this);
    return tmp;
  }

  VMCell& operator*() const
  {
    return *pos;
  }

  VMCell* operator->() const
  {
    return pos;
  }

  bool operator==(const VMCellCirculator & ccir) const
  {
    return d_mesh == ccir.d_mesh &&
	   d_c->vertex(d_s) == ccir.d_c->vertex(ccir.d_s) &&
	   d_c->vertex(d_t) == ccir.d_c->vertex(ccir.d_t) &&
	   pos == ccir.pos;
  }

  bool operator!=(const VMCellCirculator & ccir) const
  {
    return ! (*this == ccir);
  }

private:
  Mesh* d_mesh;
  VMCell* d_c;  // cell containing the considered edge
  int d_s;    // index of the source vertex of the edge in _c
  int d_t;    // index of the target vertex of the edge in _c
  VMCell* pos; // current cell
};

template<class Mesh>
class VMFacetCirculator
  : public Bidirectional_circulator_base<VMFacet, ptrdiff_t, size_t>,
    public Utilities
{
  // circulates on facets around a given edge
public:

  /*  typedef Tds_                  Tds;
  typedef typename Tds::Cell    Cell;
  typedef typename Tds::Facet   Facet;
  typedef typename Tds::Edge    Edge;
  typedef typename Tds::Vertex  Vertex;

  typedef Triangulation_ds_facet_circulator_3<Tds> Facet_circulator;
  */
  VMFacetCirculator() : d_mesh(0), d_c(0), d_s(0), d_t(0), pos(0) {}

  VMFacetCirculator(const Mesh* mesh, VMCell* c, int s, int t)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(c), d_s(s), d_t(t)
  {
    assert( c != 0 && s >= 0 && s < 4 && t >= 0 && t < 4 );
    //     if ( _tds->dimension() <3 ) useless since precondition in tds
    //     incident_facets
    pos = c;
  }

  VMFacetCirculator(const Mesh* mesh, const VMEdge & e)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(e.first), d_s(e.second), d_t(e.third)
  {
    assert( e.first != 0 && e.second >= 0 && e.second < 4 &&
	    e.third  >= 0 && e.third < 4);
    //     if ( _tds->dimension() <3 ) useless since precondition in tds
    //     incident_facets
      pos = e.first;
  }

  VMFacetCirculator(const Mesh* mesh, VMCell* c, int s, int t, VMCell* start, int f)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(c), d_s(s), d_t(t)
  {
    assert( c != 0 && s >= 0 && s < 4 && t >= 0 && t < 4 && f >= 0 && f < 4
	    && start->has_vertex( c->vertex(s) ) && start->has_vertex( c->vertex(t) ) );

    int i = start->index( c->vertex(s) );
    int j = start->index( c->vertex(t) );

    assert( f!=i && f!=j );

    if ( f == (int) next_around_edge(i,j) ) pos = start;
    else pos = start->neighbor(f); // other cell with same facet
  }

  VMFacetCirculator(const Mesh* mesh, VMCell* c, int s, int t, const VMFacet & start)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(c), d_s(s), d_t(t)
  {
    assert( c != 0 && s >= 0 && s < 4 && t >= 0 && t < 4 &&
	    start.first->has_vertex( c->vertex(s) ) &&
	    start.first->has_vertex( c->vertex(t) ) );

    int i = start.first->index( c->vertex(s) );
    int j = start.first->index( c->vertex(t) );

    assert( start.second !=i && start.second !=j );

    if ( start.second == (int) next_around_edge(i,j) ) pos = start.first;
    else pos = start.first->neighbor(start.second); // other cell with same facet
  }

  VMFacetCirculator(const Mesh* mesh, const VMEdge & e, VMCell* start, int f)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(e.first), d_s(e.second), d_t(e.third) {
    /*    assert( e.first != 0 && e.second >= 0 && e.second < 4 &&
	    e.third  >= 0 && e.third  < 4 &&  f >= 0 && f < 4 &&
	    start.first->has_vertex( e.first->vertex(e.second) ) &&
	    start.first->has_vertex( e.first->vertex(e.third) ) );
    
    int i = start.first->index( e.first->vertex(e.second) );
    int j = start.first->index( e.first->vertex(e.third) );
    
    assert( f!=i && f!=j );
    
    if ( f == (int) next_around_edge(i,j) ) pos = start.first;
    else pos = start.first->neighbor(f); // other cell with same facet*/
  }

  VMFacetCirculator(const Mesh* mesh, const VMEdge & e, const VMFacet & start)
    : d_mesh(const_cast<Mesh*>(mesh)), d_c(e.first), d_s(e.second), d_t(e.third)
  {
    assert( e.first != 0 && e.second >= 0 && e.second < 4 
	    && e.third  >= 0 && e.third  < 4 &&
	    start.first->has_vertex( e.first->vertex(e.second) ) &&
	    start.first->has_vertex( e.first->vertex(e.third) ) );

    int i = start.first->index( e.first->vertex(e.second) );
    int j = start.first->index( e.first->vertex(e.third) );

    if ( start.second == (int) next_around_edge(i,j) ) pos = start.first;
    else pos = start.first->neighbor(start.second);
  }

  VMFacetCirculator(const VMFacetCirculator & fit)
    : d_mesh(fit.d_mesh), d_c(fit.d_c), d_s(fit.d_s), d_t(fit.d_t), pos(fit.pos)
  {}

  VMFacetCirculator & operator++()
  {
    assert( pos != 0 );
    //then dimension() cannot be < 3

    pos = pos->neighbor( next_around_edge( pos->index(d_c->vertex(d_s)),
					   pos->index(d_c->vertex(d_t)) ) );
    return *this;
  }

  VMFacetCirculator operator++(int)
  {
    assert( pos != 0 );
    VMFacetCirculator tmp(*this);
    ++(*this);
    return tmp;
  }

  VMFacetCirculator & operator--()
  {
    assert( pos != 0 );

    pos = pos->neighbor( next_around_edge( pos->index(d_c->vertex(d_t)),
					   pos->index(d_c->vertex(d_s)) ) );
    return *this;
  }

  VMFacetCirculator operator--(int)
  {
    assert( pos != 0 );
    VMFacetCirculator tmp(*this);
    --(*this);
    return tmp;
  }

  VMFacet operator*() const
  {
    return std::make_pair(pos,
			  next_around_edge( pos->index(d_c->vertex(d_s)),
					    pos->index(d_c->vertex(d_t)) ) );
  }

  bool operator==(const VMFacetCirculator & fit) const
  {
    return d_mesh == fit.d_mesh &&
	   d_c->vertex(d_s) == fit.d_c->vertex(fit.d_s) &&
	   d_c->vertex(d_t) == fit.d_c->vertex(fit.d_t) &&
	   pos == fit.pos;
  }

  bool operator!=(const VMFacetCirculator & fit) const
  {
    return ! (*this == fit);
  }

private:
  Mesh* d_mesh;
  VMCell* d_c;  // cell containing the considered edge
  int d_s;    // index of the source vertex of the edge in _c
  int d_t;    // index of the target vertex of the edge in _c
  VMCell* pos; // current cell
  // the current facet is the facet of pos numbered
  // next_around_edge( pos->index(_c->vertex(_s)),
  //                   pos->index(_c->vertex(_t)) )
};

}

#endif
