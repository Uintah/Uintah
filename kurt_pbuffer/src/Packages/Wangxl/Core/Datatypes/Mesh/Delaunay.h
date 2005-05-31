#ifndef SCI_Wangxl_Datatypes_Mesh_Delaunay_h
#define SCI_Wangxl_Datatypes_Mesh_Delaunay_h

#include <iostream>
#include <set>
#include <hash_map>
#include <stack>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Defs.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/Math.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/Utilities.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/VolumeMesh.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCellBase.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertexBase.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/DCell.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DVertex.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DIterators.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DCirculators.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/BFace.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/BEdge.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/FObject.h>

namespace Wangxl {

using namespace SCIRun;

class Delaunay : public Utilities
{
  friend class DCell;
  friend class DVertex;

  friend class DCellIterator<VolumeMesh>;
  friend class DFacetIterator<VolumeMesh>;
  friend class DEdgeIterator<VolumeMesh>;
  friend class DVertexIterator<VolumeMesh>;

  friend class DCellCirculator<VolumeMesh>;
  friend class DFacetCirculator<VolumeMesh>;

protected:
  VolumeMesh d_mesh;
  DVertex* d_infinite; //infinite vertex
 
  void init_mesh() {
    d_infinite = (DVertex*) d_mesh.insert_increase_dimension(NULL);
    //    exactinit(); // initialize exact operations
  }
  
public:
  Delaunay() : d_mesh() { init_mesh(); }
  Delaunay(const Delaunay& dln) {
    d_infinite = (DVertex *) d_mesh.copy_mesh(dln.d_mesh, dln.d_infinite );
  }

  void clear() {
    d_mesh.clear();
    init_mesh();
  }

  Delaunay& operator=(const Delaunay& dln) {
    //     clear();               BUG !!
    //     infinite.Delete();
    d_infinite = (DVertex*) d_mesh.copy_mesh( dln.d_mesh, dln.d_infinite );
    return *this;
  }

  //ACCESS FUNCTIONS
  const VolumeMesh& mesh() const { return d_mesh;}
  
  int dimension() const { return d_mesh.dimension(); }
  int number_of_finite_cells() const;
  int number_of_cells() const;
  int number_of_finite_facets() const;
  int number_of_facets() const;
  int number_of_finite_edges() const;
  int number_of_edges() const;
  int number_of_vertices() const // number of finite vertices
    {return d_mesh.number_of_vertices()-1;}

  DVertex* infinite_vertex() const { return d_infinite; }
  DCell* infinite_cell() const{
    //    assert(infinite_vertex()->cell()->
    //				    has_vertex(infinite_vertex()));
    return infinite_vertex()->cell();
  }
  
  // ASSIGNMENT
  void set_number_of_vertices(int n) { d_mesh.set_number_of_vertices(n+1); }
  
  // TEST IF INFINITE FEATURES
  bool is_infinite(const DVertex* v) const 
    { return v == infinite_vertex(); }
  bool is_infinite(const DCell* c) const {
    assert( dimension() == 3 );
    return c->has_vertex(infinite_vertex());
  }
  bool is_infinite(const DCell* c, int i) const;
  bool is_infinite(const DFacet& f) const {
    return is_infinite(f.first,f.second);
  }
  bool is_infinite(const DCell* c, int i, int j) const; 
  bool is_infinite(const DEdge& e) const {
    return is_infinite(e.first,e.second,e.third);
  }


  //QUERIES

  bool is_vertex(const Point& p, DVertex* & v) const;
  bool is_vertex(DVertex* v) const;
  bool is_edge(DVertex* u, DVertex* v, DCell* c, int& i, int& j) const;
  bool is_facet(DVertex* u, DVertex* v, DVertex* w,
		DCell* c, int& i, int& j, int& k) const;
  bool is_cell(DCell* c) const;
  bool is_cell(DVertex* u, DVertex* v, DVertex* w, DVertex* t,
	       DCell* c, int& i, int& j, int& k, int& l) const;
  bool is_cell(DVertex* u, DVertex* v, DVertex* w, DVertex* t,
	       DCell* c) const;

  bool has_vertex(const DFacet& f, DVertex* v, int& j) const;
  bool has_vertex(DCell* c, int i, DVertex* v, int& j) const;
  bool has_vertex(const DFacet& f, DVertex* v) const;
  bool has_vertex(DCell* c, int i, DVertex* v) const;

  bool are_equal(DCell* c, int i, DCell* n, int j) const;
  bool are_equal(const DFacet& f, const DFacet& g) const;
  bool are_equal(const DFacet& f, DCell* n, int j) const;

  DCell* locate(const Point& p, Locate_type& lt, int& li, int& lj,
	 DCell* start = 0) const;
  DCell* locate(const Point& p, DCell* start = 0) const
  {
    Locate_type lt;
    int li, lj;
    return locate( p, lt, li, lj, start);
  }

  // PREDICATES ON POINTS ``TEMPLATED'' by the geom traits

  Bounded_side side_of_tetrahedron(const Point& p, const Point& p0, 
				   const Point& p1, const Point& p2, 
				   const Point& p3, Locate_type& lt,
				   int& i, int& j ) const;
  Bounded_side side_of_cell(const Point& p, DCell* c,
			    Locate_type& lt, int& i, int& j) const;
  Bounded_side side_of_triangle(const Point& p,
				const Point& p0, const Point& p1,
				const Point& p2,
				Locate_type& lt, int& i, int& j ) const;
  Bounded_side side_of_facet(const Point& p, DCell* c,
			     Locate_type& lt, int& li, int& lj) const;
  Bounded_side side_of_facet(const Point& p, const DFacet& f,
			     Locate_type& lt, int& li, int& lj) const {
    assert( f.second == 3 );
    return side_of_facet(p, f.first, lt, li, lj);
  }
  Bounded_side  side_of_segment(const Point& p, const Point& p0, 
				const Point& p1, Locate_type& lt, int& i ) const;
  Bounded_side side_of_edge(const Point& p, DCell* c,
			    Locate_type& lt, int& li) const;
  Bounded_side side_of_edge(const Point& p, const DEdge& e,
			    Locate_type& lt, int & li) const {
    assert( e.second == 0 );
    assert( e.third == 1 );
    return side_of_edge(p, e.first, lt, li);
  }

  //INSERTION 

  DVertex* insert_in_element(const Point& p, DCell* start = 0, DVertex* v = 0);

  DVertex* insert_in_cell(const Point& p, DCell* c, DVertex* v = 0);

  DVertex* insert_in_facet(const Point& p, DCell* c, int i, DVertex* v = 0);

  DVertex* insert_in_facet(const Point& p, const DFacet& f, DVertex* v = 0)
    {
      return insert_in_facet(p, f.first,f.second, v);
    }

  DVertex* insert_in_edge(const Point& p, DCell* c, int i, int j, DVertex* v = 0);

  DVertex* insert_in_edge(const Point& p, const DEdge& e, DVertex* v = 0)
    {
      return insert_in_edge(p, e.first,e.second,e.third, v);
    }
  
  DVertex* insert_outside_convex_hull(const Point& p, DCell* c, DVertex* v = 0);
  DVertex* insert_outside_affine_hull(const Point& p, DVertex* v = 0 );

  //Delaunay functions


  DVertex* insert(const Point& p, DCell* start = 0, DVertex* v = 0);


  DVertex* insert(const Point& p, vector<triple<VMVertex*,VMVertex*,VMVertex*> >& facets, DCell* start = 0, DVertex* v = 0);

  bool remove(DVertex* v );


  Bounded_side side_of_sphere( DCell* c, const Point& p) const;

  Bounded_side side_of_circle( const DFacet& f, const Point& p) const
    {
      return side_of_circle(f.first, f.second, p);
    }

  Bounded_side side_of_circle( DCell* c, int i, const Point & p) const;


private:

  Bounded_side side_of_sphere_inf_perturb(DVertex* v0, DVertex* v1, 
					  DVertex* v2, DVertex* v) const;

  Bounded_side side_of_sphere_finite_perturb(DVertex* v0, DVertex* v1, 
					     DVertex* v2, DVertex* v3, 
					     DVertex* v) const;
  
  int max2(int i0, int i1, int i2, int i3, int i4, int m) const;
  int maxless(int i0, int i1, int i2, int i3, int i4, int m) const;
  
  void delete_cells(std::vector<DCell*>& hole);
  
  void make_hole_3D_ear( DVertex* v, std::vector<DFacet>& boundhole,
	                 std::vector<DCell*>& hole);
  void undo_make_hole_3D_ear(std::vector<DFacet>& boundhole,
		             std::vector<DCell*>& hole);
  bool fill_hole_3D_ear(std::vector<DFacet>& boundhole);
  
private:

  class Conflict_tester_3
    {
      const Point &p;
      Delaunay *t;
    public:
      Conflict_tester_3(const Point &pt,Delaunay *dln) : p(pt), t(dln) {}
      
      bool operator()(const VMCell *c) const
	{
	  return t->side_of_sphere((DCell*)c, p) == ON_BOUNDED_SIDE;
	}
    };
  
  class Conflict_tester_2
    {
      const Point &p;
      Delaunay *t;
    public:
      
      Conflict_tester_2(const Point &pt, Delaunay *dln) : p(pt), t(dln) {}
      bool operator()(const VMCell *c) const
	{
	  return t->side_of_circle((DCell*)c, 3, p) == ON_BOUNDED_SIDE;
	}
    };

  // end of Delaunay functions

  // new added Delaunay functions

  bool get_intersected_cells(DVertex*,DVertex*,deque<DCell*>&, 
			      deque<Locate_type>[2],deque<int>[2],
			      deque<int>[2]);
  bool is_face(DVertex* v0, DVertex* v1, DVertex* v2);


public:
  bool is_edge(DVertex* v0, DVertex* v1);
  void label_cells(hash_map<Face,BFace*,FaceHash,FaceEqual>& bfaces);
  // end if new added Delaunay functions
private:
  // Here are the conflit tester function object passed to
  // d_mesh.insert_conflict() by insert_outside_convex_hull().
  class Conflict_tester_outside_convex_hull_3
  {
      const Point& p;
      Delaunay* t;

  public:

      Conflict_tester_outside_convex_hull_3(const Point &pt, Delaunay *tr)
	  : p(pt), t(tr) {}

      bool operator()(const VMCell *c) const
      {
	  Locate_type loc;
          int i, j;
	  return t->side_of_cell( p, (DCell*)c, loc, i, j )  == ON_BOUNDED_SIDE;
      }
  };

  class Conflict_tester_outside_convex_hull_2
  {
      const Point &p;
      Delaunay *t;

  public:

      Conflict_tester_outside_convex_hull_2(const Point &pt, Delaunay *tr)
	  : p(pt), t(tr) {}

      bool operator()(const VMCell *c) const
      {
	  Locate_type loc;
          int i, j;
	  return t->side_of_facet( p, (DCell*)c, loc, i, j ) == ON_BOUNDED_SIDE;
      }
  };

public:

  //TRAVERSING : ITERATORS AND CIRCULATORS
  DCellIterator<VolumeMesh> finite_cells_begin() const
    {
      if ( dimension() < 3 ) return cells_end();
      return DCellIterator<VolumeMesh>(this, false); // false means without infinite cells.
    }
  DCellIterator<VolumeMesh> all_cells_begin() const
    {
      if ( dimension() < 3 ) return cells_end();
      return DCellIterator<VolumeMesh>(this, true); // true means with infinite cells.
    }
  DCellIterator<VolumeMesh> cells_end() const
    {
      return DCellIterator<VolumeMesh>(this); // no second argument -> past-end
    }

  DVertexIterator<VolumeMesh> finite_vertices_begin() const
    {
      if ( number_of_vertices() <= 0 ) return vertices_end();
      return DVertexIterator<VolumeMesh>(this, false);
    }
  DVertexIterator<VolumeMesh> all_vertices_begin() const
    {
      if ( number_of_vertices() <= 0 ) return vertices_end();
      return DVertexIterator<VolumeMesh>(this, true);
    }
  DVertexIterator<VolumeMesh> vertices_end() const
    {
      return DVertexIterator<VolumeMesh>(this);
    }

  DEdgeIterator<VolumeMesh> finite_edges_begin() const
    {
      if ( dimension() < 1 ) return edges_end();
      return DEdgeIterator<VolumeMesh>(this, false);
    }
  DEdgeIterator<VolumeMesh> all_edges_begin() const
    {
      if ( dimension() < 1 ) return edges_end();
      return DEdgeIterator<VolumeMesh>(this, true);
    }
  DEdgeIterator<VolumeMesh> edges_end() const
    {
      return DEdgeIterator<VolumeMesh>(this);
    }

  DFacetIterator<VolumeMesh> finite_facets_begin() const
    {
      if ( dimension() < 2 ) return facets_end();
      return DFacetIterator<VolumeMesh>(this, false);
    }
  DFacetIterator<VolumeMesh> all_facets_begin() const
    {
      if ( dimension() < 2 ) return facets_end();
      return DFacetIterator<VolumeMesh>(this, true);
    }
  DFacetIterator<VolumeMesh> facets_end() const
    {
      return DFacetIterator<VolumeMesh>(this);
    }

  // cells around an edge
  DCellCirculator<VolumeMesh> incident_cells(const DEdge& e) const
    {
      assert( dimension() == 3 );
      return DCellCirculator<VolumeMesh>(this, e);
    }
  DCellCirculator<VolumeMesh> incident_cells(DCell* c, int i, int j) const
    {
      assert( dimension() == 3 );
      return DCellCirculator<VolumeMesh>(this,c,i,j);
    }
  DCellCirculator<VolumeMesh> incident_cells(const DEdge& e, DCell* start) const
    {
      assert( dimension() == 3 );
      return DCellCirculator<VolumeMesh>(this, e, start);
    }
  DCellCirculator<VolumeMesh> incident_cells(DCell* c, int i, int j, DCell* start) const  
    {
      assert( dimension() == 3 );
      return DCellCirculator<VolumeMesh>(this, c, i, j, start);
    }

  // facets around an edge
  DFacetCirculator<VolumeMesh> incident_facets(const DEdge & e) const
    {
      assert( dimension() == 3 );
      return DFacetCirculator<VolumeMesh>(this, e);
    }
  DFacetCirculator<VolumeMesh> incident_facets(DCell* c, int i, int j) const
    {
      assert( dimension() == 3 );
      return DFacetCirculator<VolumeMesh>(this, c, i, j);
    }
  DFacetCirculator<VolumeMesh> incident_facets(const DEdge& e, 
				   const DFacet& start) const
    {
      assert( dimension() == 3 );
      return DFacetCirculator<VolumeMesh>(this, e, start);
    }
  DFacetCirculator<VolumeMesh> incident_facets(DCell* c, int i, int j, 
				   const DFacet& start) const  
    {
      assert( dimension() == 3 );
      return DFacetCirculator<VolumeMesh>(this, c, i, j, start);
    }
  DFacetCirculator<VolumeMesh> incident_facets(const DEdge& e, DCell* start, int f) const
    {
      assert( dimension() == 3 );
      return DFacetCirculator<VolumeMesh>(this, e, start, f);
    }
  DFacetCirculator<VolumeMesh> incident_facets(DCell* c, int i, int j, 
				   DCell* start, int f) const  
    {
      assert( dimension() == 3 );
      return DFacetCirculator<VolumeMesh>(this, c, i, j, start, f);
    }

  // around a vertex
  void incident_cells(DVertex* v, std::set<DCell*>& cells, DCell* c = 0 ) const;

  void incident_vertices(DVertex* v, std::set<DVertex*>& vertices,
		    DCell* c = 0 ) const;

  // old methods, kept for compatibility with previous versions

private:
  void util_incident_vertices(DVertex* v, std::set<DVertex*> & vertices,
			 std::set<DCell*> & cells, DCell* c ) const;

public:

  // CHECKING
  bool is_valid(bool verbose = false, int level = 0) const;

  bool is_valid_finite(DCell* c, bool verbose = false, int level = 0) const;
};

int Delaunay::number_of_finite_cells() const 
{
  if ( dimension() < 3 ) return 0;
  return std::distance(finite_cells_begin(), cells_end());
}
  
int Delaunay::number_of_cells() const 
{
  if ( dimension() < 3 ) return 0;
  return std::distance(all_cells_begin(), cells_end());
}

int Delaunay::number_of_finite_facets() const
{
  if ( dimension() < 2 ) return 0;
  return std::distance(finite_facets_begin(), facets_end());
}

int Delaunay::number_of_facets() const
{
  if ( dimension() < 2 ) return 0;
  return std::distance(all_facets_begin(), facets_end());
}

int Delaunay::number_of_finite_edges() const
{
  if ( dimension() < 1 ) return 0;
  return std::distance(finite_edges_begin(), edges_end());
}

int Delaunay::number_of_edges() const
{
  if ( dimension() < 1 ) return 0;
  return std::distance(all_edges_begin(), edges_end());
}

bool Delaunay::is_infinite(const DCell* c, int i) const 
{
  assert( dimension() == 2 || dimension() == 3 );
  assert( (dimension() == 2 && i == 3)
	  || (dimension() == 3 && i >= 0 && i <= 3) );
  return is_infinite(c->vertex(i<=0 ? 1 : 0)) ||
	 is_infinite(c->vertex(i<=1 ? 2 : 1)) ||
	 is_infinite(c->vertex(i<=2 ? 3 : 2));
}

bool Delaunay::is_infinite(const DCell* c, int i, int j) const 
{ 
  assert( i != j );
  assert( dimension() >= 1 && dimension() <= 3 );
  assert( i >= 0 && i <= dimension() && j >= 0 && j <= dimension() );
  return is_infinite( c->vertex(i) ) || is_infinite( c->vertex(j) );
}

bool Delaunay::is_vertex(const Point & p, DVertex* & v) const
{
  Locate_type lt;
  int li, lj;
  DCell* c = locate( p, lt, li, lj );
  if ( lt != VERTEX )
    return false;
  v = c->vertex(li);
  return true;
}

bool Delaunay::is_vertex(DVertex* v) const
{
  return d_mesh.is_vertex(v);
}

bool Delaunay::is_edge(DVertex* u, DVertex* v, DCell* c, int& i, int& j) const
{
  assert( d_mesh.is_vertex(u) && d_mesh.is_vertex(v) );
  VMCell* cstar;
  bool b = d_mesh.is_edge(u, v, cstar, i, j);
  if (b)  c = (DCell*) cstar;
  return b;
}

bool Delaunay::is_facet(DVertex* u, DVertex* v, DVertex* w,
	 DCell* c, int& i, int& j, int& k) const
{
  assert( d_mesh.is_vertex(u) && d_mesh.is_vertex(v) &&
	  d_mesh.is_vertex(w) );
  VMCell* cstar;
  bool b = d_mesh.is_facet(u, v, w, cstar, i, j, k);
  if (b)  c = (DCell*) cstar;
  return b;
}

inline bool Delaunay::is_cell(DCell* c) const
{
  return d_mesh.is_cell(c);
}

bool Delaunay::is_cell(DVertex* u, DVertex* v, DVertex* w, DVertex* t,
	DCell* c, int& i, int& j, int& k, int& l) const
{
  assert( d_mesh.is_vertex(u) && d_mesh.is_vertex(v) 
	  && d_mesh.is_vertex(w) && d_mesh.is_vertex(t) );
  VMCell* cstar;
  bool b = d_mesh.is_cell(u, v, w, t, cstar, i, j, k, l);
  if (b) c = (DCell*)cstar;
  return b;
}

bool Delaunay::is_cell(DVertex* u, DVertex* v, DVertex* w, DVertex* t,
	DCell* c) const
{
  assert( d_mesh.is_vertex(u) && d_mesh.is_vertex(v) 
	  && d_mesh.is_vertex(w) && d_mesh.is_vertex(t) );
  int i,j,k,l;
  VMCell* cstar;
  bool b = d_mesh.is_cell(u, v, w, t, cstar, i, j, k, l);
  if (b) c = (DCell*)cstar;
  return b;
}

inline bool Delaunay::
has_vertex(const DFacet& f, DVertex* v, int& j) const
{
  return d_mesh.has_vertex(f.first, f.second, v, j);
}

inline bool Delaunay::
has_vertex(DCell* c, int i, DVertex* v, int& j) const
{
  return d_mesh.has_vertex(c, i, v, j);
}

inline bool Delaunay::has_vertex(const DFacet& f, DVertex* v) const
{
  return d_mesh.has_vertex(f.first, f.second, v);
}

inline bool Delaunay::has_vertex(DCell* c, int i, DVertex* v) const
{
  return d_mesh.has_vertex(&*c, i, &*v);
}

inline bool Delaunay::are_equal(DCell* c, int i, DCell* n, int j) const
{
  return d_mesh.are_equal(c, i, n, j);
}

inline bool Delaunay::are_equal(const DFacet& f, const DFacet& g) const
{
  return d_mesh.are_equal(f.first, f.second, g.first, g.second);
}

inline bool Delaunay::are_equal(const DFacet& f, DCell* n, int j) const
{
  return d_mesh.are_equal(f.first, f.second, n, j);
}

DCell* Delaunay::
locate(const Point& p, Locate_type& lt, int& li, int& lj,
       DCell* start ) const
  // returns the (finite or infinite) cell p lies in
  // starts at cell "start"
  // start must be finite
  // if lt == OUTSIDE_CONVEX_HULL, li is the
  // index of a facet separating p from the rest of the triangulation
  // in dimension 2 :
  // returns a facet (DCell*,li) if lt == FACET
  // returns an edge (DCell*,li,lj) if lt == EDGE
  // returns a vertex (DCell*,li) if lt == VERTEX
  // if lt == OUTSIDE_CONVEX_HULL, li, lj give the edge of c
  // separating p from the rest of the triangulation
  // lt = OUTSIDE_AFFINE_HULL if p is not coplanar with the triangulation
{
  int i, inf;

  if ( dimension() >= 1 && start == 0 )
    // there is at least one finite "cell" (or facet or edge)
    start = infinite_vertex()->cell()->neighbor
            ( infinite_vertex()->cell()->index( infinite_vertex()) );

  switch (dimension()) {
  case 3:
    {
      assert( start != 0 );
      DCell *c, *previous;
      int ind_inf;
      if ( start->has_vertex(d_infinite, ind_inf) )
	c = start->neighbor(ind_inf);
      else
	c = start;
 
      Orientation o[4];

      // We implement the remembering visibility/stochastic walk.

      // Main locate loop
      while(1) {
	if ( c->has_vertex(d_infinite,li) ) {
	  // c must contain p in its interior
	  lt = OUTSIDE_CONVEX_HULL;
	  return c;
	}

	// FIXME: do more benchmarks.
	i = rand_4(); // For the (remembering) stochastic walk
	// i = 0; // For the (remembering) visibility walk. Ok for Delaunay only

        Orientation test_or = (i&1)==0 ? NEGATIVE : POSITIVE;
	const Point& p0 = c->vertex( i )->point();
	const Point& p1 = c->vertex( (i+1)&3 )->point();
	const Point& p2 = c->vertex( (i+2)&3 )->point();
	const Point& p3 = c->vertex( (i+3)&3 )->point();

	// Note : among the four Points, 3 are common with the previous cell...
	// Something can probably be done to take advantage of this, like
	// storing the four in an array and changing only one ?

	// We could make a loop of these 4 blocks, for clarity, but not speed.
	DCell* next = c->neighbor(i);
	if (previous != next) {
	  o[0] = orientation(p, p1, p2, p3);
	  if ( o[0] == test_or) {
	    previous = c;
	    c = next;
	    continue;
	  }
	}
	else o[0] = (Orientation) - test_or;

	next = c->neighbor((i+1)&3);
	if (previous != next) {
	  o[1] = orientation(p0, p, p2, p3);
	  if ( o[1] == test_or) {
	    previous = c;
	    c = next;
	    continue;
	  }
	} 
	else o[1] = (Orientation) - test_or;

	next = c->neighbor((i+2)&3);
	if (previous != next) {
	  o[2] = orientation(p0, p1, p, p3);
	  if ( o[2] == test_or) {
	    previous = c;
	    c = next;
	    continue;
	  }
	}
	else o[2] = (Orientation) - test_or;

	next = c->neighbor((i+3)&3);
	if (previous != next) {
	  o[3] = orientation(p0, p1, p2, p);
	  if ( o[3] == test_or) {
	    // previous = c; // not necessary because it's the last one.
	    c = next;
	    continue;
	  }
	} 
	else o[3] = (Orientation) - test_or;

	break;
      }
      
      // now p is in c or on its boundary
      int sum = ( o[0] == COPLANAR )
	+ ( o[1] == COPLANAR )
	+ ( o[2] == COPLANAR )
	+ ( o[3] == COPLANAR );
      switch (sum) {
      case 0:
	{
	  lt = CELL;
	  break;
	}
      case 1:
	{ 
	  lt = FACET;
	  li = ( o[0] == COPLANAR ) ? i :
	    ( o[1] == COPLANAR ) ? (i+1)&3 :
	    ( o[2] == COPLANAR ) ? (i+2)&3 :
	    (i+3)&3;
	  break;
	}
      case 2:
	{ 
	  lt = EDGE;
	  li = ( o[0] != COPLANAR ) ? i :
	    ( o[1] != COPLANAR ) ? ((i+1)&3) :
	    ((i+2)&3);
	  lj = ( o[ (li+1-i)&3 ] != COPLANAR ) ? ((li+1)&3) :
	    ( o[ (li+2-i)&3 ] != COPLANAR ) ? ((li+2)&3) :
	    ((li+3)&3);
	  assert(collinear( p, c->vertex( li )->point(),
			    c->vertex( lj )->point() ));
	  break;
	}
      case 3:
	{
	  lt = VERTEX;
	  li = ( o[0] != COPLANAR ) ? i :
	    ( o[1] != COPLANAR ) ? (i+1)&3 :
	    ( o[2] != COPLANAR ) ? (i+2)&3 :
	    (i+3)&3;
	  break;
	}
      }
      return c;
    }
  case 2:
    {
      assert( start != 0 );
      DCell* c;
      int ind_inf;
      if ( start->has_vertex(d_infinite, ind_inf) )
	c = start->neighbor(ind_inf);
      else
	c = start;

      //first tests whether p is coplanar with the current triangulation
      DFacetIterator<VolumeMesh> finite_fit = finite_facets_begin();
      if ( orientation( (*finite_fit).first->vertex(0)->point(),
			(*finite_fit).first->vertex(1)->point(),
			(*finite_fit).first->vertex(2)->point(),
			p ) != DEGENERATE ) {
	lt = OUTSIDE_AFFINE_HULL;
	li = 3; // only one facet in dimension 2
	return (*finite_fit).first;
      }
      // if p is coplanar, location in the triangulation
      // only the facet numbered 3 exists in each cell
      while (1) {
	  
	if ( c->has_vertex(d_infinite,inf) ) {
	  // c must contain p in its interior
	  lt = OUTSIDE_CONVEX_HULL;
	  li = cw(inf);
	  lj = ccw(inf);
	  return c;
	}

	// else c is finite
	// we test its edges in a random order until we find a
	// neighbor to go further
	i = rand_3();
	const Point & p0 = c->vertex( i )->point();
	const Point & p1 = c->vertex( ccw(i) )->point();
	const Point & p2 = c->vertex( cw(i) )->point();
        Orientation o[3];
	assert(orientation(p0,p1,p2)==POSITIVE); // colinear test
	o[0] = orientation(p0,p1,p);
	if ( o[0] == NEGATIVE ) {
	  c = c->neighbor( cw(i) );
	  continue;
	}
	o[1] = orientation(p1,p2,p);
	if ( o[1] == NEGATIVE ) {
	  c = c->neighbor( i );
	  continue;
	}
	o[2] = orientation(p2,p0,p);
	if ( o[2] == NEGATIVE ) {
	  c = c->neighbor( ccw(i) );
	  continue;
	}

	// now p is in c or on its boundary
	int sum = ( o[0] == COLLINEAR )
	        + ( o[1] == COLLINEAR )
	        + ( o[2] == COLLINEAR );
	switch (sum) {
	case 0:
	  {
	    lt = FACET;
	    li = 3; // useless ?
	    break;
	  }
	case 1:
	  {
	    lt = EDGE;
	    li = ( o[0] == COLLINEAR ) ? i :
	         ( o[1] == COLLINEAR ) ? ccw(i) :
	         cw(i);
	    lj = ccw(li);
	    break;
	  }
	case 2:
	  {
	    lt = VERTEX;
	    li = ( o[0] != COLLINEAR ) ? cw(i) :
	         ( o[1] != COLLINEAR ) ? i :
	         ccw(i);
	    break;
	  }
	}
	return c;
      }
    }
  case 1:
    {
      assert( start != 0 );
      DCell* c;
      int ind_inf;
      if ( start->has_vertex(d_infinite, ind_inf) )
	c = start->neighbor(ind_inf);
      else
	c = start;

      //first tests whether p is collinear with the current triangulation
      DEdgeIterator<VolumeMesh> finite_eit = finite_edges_begin();
      if ( ! collinear( p,(*finite_eit).first->vertex(0)->point(),
			(*finite_eit).first->vertex(1)->point()) ) {
	lt = OUTSIDE_AFFINE_HULL;
	return (*finite_eit).first;
      }
      // if p is collinear, location :
      Comparison_result o, o0, o1;
      int xyz;
      Point p0 = c->vertex(0)->point();
      Point p1 = c->vertex(1)->point();
      assert( ( compare_x(p0,p1) != EQUAL ) ||
	      ( compare_y(p0,p1) != EQUAL ) ||
	      ( compare_z(p0,p1) != EQUAL ) );
      o = compare_x(p0,p1);
      if ( o == EQUAL ) {
	o = compare_y(p0,p1);
	if ( o == EQUAL ) {
	  o = compare_z(p0,p1);
	  xyz = 3;
	}
	else 
	  xyz = 2;
      }
      else 
	xyz  = 1;
      //	bool notfound = true;
      while (1) {
	if ( c->has_vertex(d_infinite,inf) ) {
	  // c must contain p in its interior
	  lt = OUTSIDE_CONVEX_HULL;
	  return c;
	}

	// else c is finite
	// we test on which direction to continue the traversal
	p0 = c->vertex(0)->point();
	p1 = c->vertex(1)->point();
	switch ( xyz ) {
	case 1:
	  {
	    o = compare_x(p0,p1);
	    o0 = compare_x(p0,p);
	    o1 = compare_x(p,p1);
	    break;
	  }
	case 2:
	  {
	    o = compare_y(p0,p1);
	    o0 = compare_y(p0,p);
	    o1 = compare_y(p,p1);
	    break;
	  }
	default: // case 3
	  {
	    o = compare_z(p0,p1);
	    o0 = compare_z(p0,p);
	    o1 = compare_z(p,p1);
	  }
	}
	  
	if (o0 == EQUAL) {
	  lt = VERTEX;
	  li = 0;
	  return c;
	}
	if (o1 == EQUAL) {
	  lt = VERTEX;
	  li = 1;
	  return c;
	}
	if ( o0 == o1 ) {
	  lt = EDGE;
	  li = 0;
	  lj = 1;
	  return c;
	}
	if ( o0 == o ) { 
	  c = c->neighbor(0);
	  continue;
	}
	if ( o1 == o ) { 
	  c = c->neighbor(1);
	  continue; 
	}
      }
    }
  case 0:
    {
      DVertexIterator<VolumeMesh> vit = finite_vertices_begin();
      if ( ! equal( p, vit->point() ) ) {
	lt = OUTSIDE_AFFINE_HULL;
      }
      else {
	lt = VERTEX;
	li = 0;
      }
      return vit->cell();
    }
  case -1:
    {
      lt = OUTSIDE_AFFINE_HULL;
      return 0;
    }
  default:
    {
      assert(false);
      return 0;
    }
  }
}
	  
Bounded_side Delaunay::
side_of_tetrahedron(const Point & p, const Point & p0, const Point & p1, 
		    const Point & p2, const Point & p3, Locate_type& lt, 
		    int & i, int & j ) const
  // p0,p1,p2,p3 supposed to be non coplanar
  // tetrahedron p0,p1,p2,p3 is supposed to be well oriented
  // returns :
  // ON_BOUNDED_SIDE if p lies strictly inside the tetrahedron
  // ON_BOUNDARY if p lies on one of the facets
  // ON_UNBOUNDED_SIDE if p lies strictly outside the tetrahedron
{
  assert( orientation(p0,p1,p2,p3) == POSITIVE );

  Orientation o0,o1,o2,o3;
  if ( ((o0 = orientation(p,p1,p2,p3)) == NEGATIVE) ||
       ((o1 = orientation(p0,p,p2,p3)) == NEGATIVE) ||
       ((o2 = orientation(p0,p1,p,p3)) == NEGATIVE) ||
       ((o3 = orientation(p0,p1,p2,p)) == NEGATIVE) ) {
    lt = OUTSIDE_CONVEX_HULL;
    return ON_UNBOUNDED_SIDE;
  }

  // now all the oi's are >=0
  // sum gives the number of facets p lies on
  int sum = ( (o0 == ZERO) ? 1 : 0 ) 
          + ( (o1 == ZERO) ? 1 : 0 ) 
          + ( (o2 == ZERO) ? 1 : 0 ) 
          + ( (o3 == ZERO) ? 1 : 0 );

  switch (sum) {
  case 0:
    {
      lt = CELL;
      return ON_BOUNDED_SIDE;
    }
  case 1:
    {
      lt = FACET;
      // i = index such that p lies on facet(i)
      i = ( o0 == ZERO ) ? 0 :
	  ( o1 == ZERO ) ? 1 :
	  ( o2 == ZERO ) ? 2 :
	  3;
      return ON_BOUNDARY;
    }
  case 2:
    {
      lt = EDGE;
      // i = smallest index such that p does not lie on facet(i)
      // i must be < 3 since p lies on 2 facets
      i = ( o0 == POSITIVE ) ? 0 :
	  ( o1 == POSITIVE ) ? 1 :
	  2;
      // j = larger index such that p not on facet(j)
      // j must be > 0 since p lies on 2 facets
      j = ( o3 == POSITIVE ) ? 3 :
	  ( o2 == POSITIVE ) ? 2 :
	  1;
      return ON_BOUNDARY;
    }
  case 3:
    {
      lt = VERTEX;
      // i = index such that p does not lie on facet(i)
      i = ( o0 == POSITIVE ) ? 0 :
	  ( o1 == POSITIVE ) ? 1 :
	  ( o2 == POSITIVE ) ? 2 :
	  3;
      return ON_BOUNDARY;
    }
  default:
    {
      // impossible : cannot be on 4 facets for a real tetrahedron
      assert(false);
      return ON_BOUNDARY;
    }
  }
}

Bounded_side Delaunay::side_of_cell(const Point& p, DCell* c,
	     Locate_type& lt, int& i, int& j) const
  // returns
  // ON_BOUNDED_SIDE if p inside the cell
  // (for an infinite cell this means that p lies strictly in the half space
  // limited by its finite facet)
  // ON_BOUNDARY if p on the boundary of the cell
  // (for an infinite cell this means that p lies on the *finite* facet)
  // ON_UNBOUNDED_SIDE if p lies outside the cell
  // (for an infinite cell this means that p is not in the preceding
  // two cases)  
  // lt has a meaning only when ON_BOUNDED_SIDE or ON_BOUNDARY
{
  assert( dimension() == 3 );
  if ( ! is_infinite(c) ) {
    return side_of_tetrahedron(p, c->vertex(0)->point(),
			       c->vertex(1)->point(),
			       c->vertex(2)->point(),
			       c->vertex(3)->point(), lt, i, j);
  }
  else {
    int inf = c->index(d_infinite);
    Orientation o;
    DVertex* v1=c->vertex((inf+1)&3);
    DVertex* v2=c->vertex((inf+2)&3); 
    DVertex* v3=c->vertex((inf+3)&3);
    if ( (inf&1) == 0 ) 
      o = orientation(p, v1->point(), v2->point(), v3->point());
    else 
      o =  orientation(v3->point(), p, v1->point(), v2->point());

    switch (o) {
    case POSITIVE:
      {
	lt = CELL;
	return ON_BOUNDED_SIDE;
      }
    case NEGATIVE:
      return ON_UNBOUNDED_SIDE;
    case ZERO:
      {
	// location in the finite facet
	int i_f, j_f;
	Bounded_side side = 
	  side_of_triangle(p, v1->point(), v2->point(), v3->point(),
			   lt, i_f, j_f);
	// lt need not be modified in most cases :
	switch (side) {
	case ON_BOUNDED_SIDE:
	  {
	    // lt == FACET ok
	    i = inf;
	    return ON_BOUNDARY;
	  }
	case ON_BOUNDARY:
	  {
	    // lt == VERTEX OR EDGE ok
	    i = ( i_f == 0 ) ? ((inf+1)&3) :
	        ( i_f == 1 ) ? ((inf+2)&3) :
	        ((inf+3)&3);
	    if ( lt == EDGE ) {
	      j = (j_f == 0 ) ? ((inf+1)&3) :
		  ( j_f == 1 ) ? ((inf+2)&3) :
		  ((inf+3)&3);
	    }
	    return ON_BOUNDARY;
	  }
	case ON_UNBOUNDED_SIDE:
	  {
	    // p lies on the plane defined by the finite facet
	    // lt must be initialized
	    return ON_UNBOUNDED_SIDE;
	  }
	default:
	  {
	    assert(false);
	    return ON_BOUNDARY;
	  }
	} // switch side
      }// case ZERO
    default:
      {
	assert(false);
	return ON_BOUNDARY;
      }
    } // switch o
  } // else infinite cell
} // side_of_cell

Bounded_side Delaunay::
side_of_triangle(const Point & p, const Point & p0, 
		 const Point & p1, const Point & p2,
		 Locate_type & lt, int & i, int & j ) const
  // p0,p1,p2 supposed to define a plane
  // p supposed to lie on plane p0,p1,p2
  // triangle p0,p1,p2 defines the orientation of the plane
  // returns
  // ON_BOUNDED_SIDE if p lies strictly inside the triangle
  // ON_BOUNDARY if p lies on one of the edges
  // ON_UNBOUNDED_SIDE if p lies strictly outside the triangle
{
  assert( orientation(p,p0,p1,p2) == COPLANAR );

  Orientation o012 = orientation(p0,p1,p2);
  assert( o012 != COLLINEAR );

  Orientation o0; // edge p0 p1
  Orientation o1; // edge p1 p2
  Orientation o2; // edge p2 p0

  if ((o0 = orientation(p0,p1,p)) == opposite(o012) ||
      (o1 = orientation(p1,p2,p)) == opposite(o012) ||
      (o2 = orientation(p2,p0,p)) == opposite(o012)) {
    lt = OUTSIDE_CONVEX_HULL;
    return ON_UNBOUNDED_SIDE;
  }

  // now all the oi's are >=0
  // sum gives the number of edges p lies on
  int sum = ( (o0 == ZERO) ? 1 : 0 ) 
          + ( (o1 == ZERO) ? 1 : 0 ) 
          + ( (o2 == ZERO) ? 1 : 0 );

  switch (sum) {
  case 0:
    {
      lt = FACET;
      return ON_BOUNDED_SIDE;
    }
  case 1:
    {
      lt = EDGE;
      i = ( o0 == ZERO ) ? 0 :
	  ( o1 == ZERO ) ? 1 :
	  2;
      if ( i == 2 ) 
	j=0;
      else 
	j = i+1;
      return ON_BOUNDARY;
    }
  case 2:
    {
      lt = VERTEX;
      i = ( o0 == o012 ) ? 2 :
	  ( o1 == o012 ) ? 0 :
	  1;
      return ON_BOUNDARY;
    }
  default:
    {
      // cannot happen
      assert(false);
      return ON_BOUNDARY;
    }
  }
}

Bounded_side Delaunay::side_of_facet(const Point & p, DCell* c,
	      Locate_type & lt, int & li, int & lj) const
  // supposes dimension 2 otherwise does not work for infinite facets
  // returns :
  // ON_BOUNDED_SIDE if p inside the facet
  // (for an infinite facet this means that p lies strictly in the half plane
  // limited by its finite edge)
  // ON_BOUNDARY if p on the boundary of the facet
  // (for an infinite facet this means that p lies on the *finite* edge)
  // ON_UNBOUNDED_SIDE if p lies outside the facet
  // (for an infinite facet this means that p is not in the
  // preceding two cases) 
  // lt has a meaning only when ON_BOUNDED_SIDE or ON_BOUNDARY
  // when they mean anything, li and lj refer to indices in the cell c 
  // giving the facet (c,i)
{
  assert( dimension() == 2 );
  if ( ! is_infinite(c,3) ) {
    // The following precondition is useless because it is written
    // in side_of_facet  
    // 	assert( orientation (p, 
    // 					  c->vertex(0)->point,
    // 					  c->vertex(1)->point,
    // 					  c->vertex(2)->point) == COPLANAR );
    int i_t, j_t;
    Bounded_side side = side_of_triangle(p,
			    c->vertex(0)->point(),
			    c->vertex(1)->point(),
			    c->vertex(2)->point(),
			    lt, i_t, j_t);
    // indices in the original cell :
    li = ( i_t == 0 ) ? 0 :
         ( i_t == 1 ) ? 1 : 2;
    lj = ( j_t == 0 ) ? 0 :
         ( j_t == 1 ) ? 1 : 2;
    return side;
  }
  // else infinite facet
  int inf = c->index(d_infinite);
    // The following precondition is useless because it is written
    // in side_of_facet  
    // 	assert( orientation (p,
    // 					  c->neighbor(inf)->vertex(0)->point(),
    // 					  c->neighbor(inf)->vertex(1)->point(),
    // 					  c->neighbor(inf)->vertex(2)->point())
    // 					 == COPLANAR );
  int i2 = next_around_edge(inf,3);
  int i1 = 3-inf-i2;
  DVertex* v1 = c->vertex(i1);
  DVertex* v2 = c->vertex(i2);

  assert(orientation(v1->point(), v2->point(),
		     c->mirror_vertex(inf)->point()) == POSITIVE);
  Orientation o = orientation(v1->point(), v2->point(), p);
  switch (o) {
  case POSITIVE:
    // p lies on the same side of v1v2 as vn, so not in f
    {
      return ON_UNBOUNDED_SIDE;
    }
  case NEGATIVE:
    // p lies in f
    { 
      lt = FACET;
      li = 3;
      return ON_BOUNDED_SIDE;
    }
  case ZERO:
    // p collinear with v1v2
    {
      int i_e;
      switch (side_of_segment(p, v1->point(), v2->point(), lt, i_e)) {
	// computation of the indices in the original cell
      case ON_BOUNDED_SIDE:
	{
	  // lt == EDGE ok
	  li = i1;
	  lj = i2;
	  return ON_BOUNDARY;
	}
      case ON_BOUNDARY:
	{
	  // lt == VERTEX ok
	  li = ( i_e == 0 ) ? i1 : i2;
	  return ON_BOUNDARY;
	}
      case ON_UNBOUNDED_SIDE:
	{
	  // p lies on the line defined by the finite edge
	  return ON_UNBOUNDED_SIDE;
	}
      default:
	{
	  // cannot happen. only to avoid warning with eg++
	  return ON_UNBOUNDED_SIDE;
	}
      } 
    }// case ZERO
  }// switch o
  // end infinite facet
  // cannot happen. only to avoid warning with eg++
  assert(false);
  return ON_UNBOUNDED_SIDE;
}

Bounded_side Delaunay::
side_of_segment(const Point & p, const Point & p0, const Point & p1,
		Locate_type & lt, int & i ) const
  // p0, p1 supposed to be different
  // p supposed to be collinear to p0, p1
  // returns :
  // ON_BOUNDED_SIDE if p lies strictly inside the edge
  // ON_BOUNDARY if p equals p0 or p1
  // ON_UNBOUNDED_SIDE if p lies strictly outside the edge
{
  assert( ! equal(p0, p1) );
  assert( collinear(p, p0, p1) );
      
  Comparison_result c = compare_x(p0,p1);
  Comparison_result c0;
  Comparison_result c1;

  if ( c == EQUAL ) {
    c = compare_y(p0,p1);
    if ( c == EQUAL ) {
      c0 = compare_z(p0,p);
      c1 = compare_z(p,p1);
    }
    else {
      c0 = compare_y(p0,p);
      c1 = compare_y(p,p1);
    }
  }
  else {
    c0 = compare_x(p0,p);
    c1 = compare_x(p,p1);
  }
      
  //      if ( (c0 == SMALLER) && (c1 == SMALLER) ) {
  if ( c0 == c1 ) {
    lt = EDGE;
    return ON_BOUNDED_SIDE;
  }
  if (c0 == EQUAL) {
    lt = VERTEX;
    i = 0;
    return ON_BOUNDARY;
  }
  if (c1 == EQUAL) {
    lt = VERTEX;
    i = 1;
    return ON_BOUNDARY;
  }
  lt = OUTSIDE_CONVEX_HULL;
  return ON_UNBOUNDED_SIDE;
}

Bounded_side Delaunay::side_of_edge(const Point & p, DCell* c,
	     Locate_type & lt, int & li) const
  // supposes dimension 1 otherwise does not work for infinite edges
  // returns :
  // ON_BOUNDED_SIDE if p inside the edge 
  // (for an infinite edge this means that p lies in the half line
  // defined by the vertex)
  // ON_BOUNDARY if p equals one of the vertices
  // ON_UNBOUNDED_SIDE if p lies outside the edge
  // (for an infinite edge this means that p lies on the other half line)
  // lt has a meaning when ON_BOUNDED_SIDE and ON_BOUNDARY  
  // li refer to indices in the cell c 
{//side_of_edge
  assert( dimension() == 1 );
  if ( ! is_infinite(c,0,1) ) 
    return side_of_segment(p, c->vertex(0)->point(), c->vertex(1)->point(),
			   lt, li);
  // else infinite edge
  int inf = c->index(d_infinite);
  if ( equal( p, c->vertex(1-inf)->point() ) ) {
    lt = VERTEX;
    li = 1-inf;
    return ON_BOUNDARY;
  }
  // does not work in dimension > 2
  DCell* n = c->neighbor(inf);
  int i_e = n->index(c);
  // we know that n is finite
  DVertex* v0 = n->vertex(0);
  DVertex* v1 = n->vertex(1);
  Comparison_result c01 = compare_x(v0->point(), v1->point());
  Comparison_result cp;
  if ( c01 == EQUAL ) {
    c01 = compare_y(v0->point(),v1->point());
    if ( i_e == 0 ) {
      cp = compare_y( v1->point(), p );
    }
    else {
      cp = compare_y( p, v0->point() );
    }
  }
  else {
    if ( i_e == 0 ) 
      cp = compare_x( v1->point(), p );
    else 
      cp = compare_x( p, v0->point() );
  }
  if ( c01 == cp ) {
    // p lies on the same side of n as infinite
    lt = EDGE;
    return ON_BOUNDED_SIDE;
  }
  return ON_UNBOUNDED_SIDE;
}

DVertex* Delaunay::insert_in_element(const Point& p, DCell* start, DVertex* v)
{
  Locate_type lt;
  int li, lj;
  DCell* c = locate( p, lt, li, lj, start);
  switch (lt) {
  case VERTEX:
    return c->vertex(li);
  case EDGE:
    return insert_in_edge(p, c, li, lj, v);
  case FACET:
    return insert_in_facet(p, c, li, v);
  case CELL:
    return insert_in_cell(p, c, v);
  case OUTSIDE_CONVEX_HULL:
    return insert_outside_convex_hull(p, c, v);
  case OUTSIDE_AFFINE_HULL:
  default:
    return insert_outside_affine_hull(p, v);
  }
}

DVertex* Delaunay::insert_in_cell(const Point& p, DCell* c, DVertex* v)
{
  assert( dimension() == 3 );
  Locate_type lt;
  int i; 
  int j;
  assert( side_of_tetrahedron( p, c->vertex(0)->point(),
			       c->vertex(1)->point(),
			       c->vertex(2)->point(),
			       c->vertex(3)->point(),
			       lt,i,j ) == ON_BOUNDED_SIDE );

    v = (DVertex*)d_mesh.insert_in_cell(v, c);
    v->set_point(p);
    return v;
}

DVertex* Delaunay::
insert_in_facet(const Point& p, DCell* c, int i, DVertex* v)
{
  assert( dimension() == 2 || dimension() == 3);
  assert( (dimension() == 2 && i == 3)
	  || (dimension() == 3 && i >= 0 && i <= 3) );
  Locate_type lt;
  int li; int lj;
  assert( orientation( p, c->vertex((i+1)&3)->point(),
		       c->vertex((i+2)&3)->point(),
		       c->vertex((i+3)&3)->point() ) == COPLANAR
	  && side_of_triangle( p, c->vertex((i+1)&3)->point(),
			       c->vertex((i+2)&3)->point(),
			       c->vertex((i+3)&3)->point(),
			       lt, li, lj) == ON_BOUNDED_SIDE );
  v = (DVertex*) d_mesh.insert_in_facet( v, c, i);
  v->set_point(p);
  return v;
}

DVertex* Delaunay::
insert_in_edge(const Point& p, DCell* c, int i, int j, DVertex* v)
{
  assert( i != j );
  assert( dimension() >= 1 && dimension() <= 3 );
  assert( i >= 0 && i <= dimension() && j >= 0 && j <= dimension() );
  Locate_type lt; int li;
  switch ( dimension() ) {
  case 3:
  case 2:
    {
      assert( ! is_infinite(c, i, j) );
      assert( collinear( c->vertex(i)->point(), p, c->vertex(j)->point() )
	      && side_of_segment( p, c->vertex(i)->point(),
				  c->vertex(j)->point(),lt, li ) == ON_BOUNDED_SIDE );
      break;
    }
  case 1:
    {
      assert( side_of_edge(p, c, lt, li) == ON_BOUNDED_SIDE );
      break;
    }
  }

  v = (DVertex*) d_mesh.insert_in_edge( v, c, i, j);
  v->set_point(p);
  return v;
}

DVertex* Delaunay::insert_outside_convex_hull(const Point& p, DCell* c, DVertex* v)
  // c is an infinite cell containing p
  // p is strictly outside the convex hull
  // dimension 0 not allowed, use outside-affine-hull
{
  assert( dimension() > 0 );
  assert( c->has_vertex(d_infinite) );
  // the precondition that p is in c is tested in each of the
  // insertion methods called from this method 
  switch ( dimension() ) {
  case 1:
    {
      // 	// p lies in the infinite edge neighboring c 
      // 	// on the other side of li
      // 	return insert_in_edge(p,c->neighbor(1-li),0,1);
      return insert_in_edge(p,c,0,1,v);
    }
  case 2:
    {
      set_number_of_vertices(number_of_vertices()+1);

      Conflict_tester_outside_convex_hull_2 tester(p, this);
      DVertex* v = (DVertex*) d_mesh.insert_conflict(0, c, tester);
      v->set_point(p);
      
      return v;
    }
  case 3:
  default:
    {
      set_number_of_vertices(number_of_vertices()+1);

      Conflict_tester_outside_convex_hull_3 tester(p, this);
      DVertex* v = (DVertex*) d_mesh.insert_conflict(0, c, tester);
      v->set_point(p);
      return v;
    }
  }
}

DVertex* Delaunay::insert_outside_affine_hull(const Point& p, DVertex* v)
{
  assert( dimension() < 3 );
  bool reorient;
  switch ( dimension() ) {
  case 1:
    {
      DCell* c = infinite_cell();
      DCell* n = c->neighbor(c->index(infinite_vertex()));
      Orientation o = orientation(n->vertex(0)->point(),
					   n->vertex(1)->point(), p);
      assert ( o != COLLINEAR );
      reorient = o == NEGATIVE;
      break;
    }
  case 2:
    {
      DCell* c = infinite_cell();
      DCell* n = c->neighbor(c->index(infinite_vertex()));
      Orientation o = orientation( n->vertex(0)->point(),
			           n->vertex(1)->point(),
			           n->vertex(2)->point(), p );
      assert ( o != COPLANAR );
      reorient = o == NEGATIVE;
      break;
    }
  default:
    reorient = false;
  }

  v = (DVertex*) d_mesh.insert_increase_dimension( (VMVertex*)v, (VMVertex*)infinite_vertex(), reorient);
  v->set_point(p);
  return v;
}

void Delaunay::incident_cells(DVertex* v, std::set<DCell*>& cells, DCell* c ) const
{
  assert( v != 0 );
  assert( d_mesh.is_vertex(v) );

  if ( dimension() < 3 )  return;

  if ( c == 0 ) c = v->cell();
  else  assert( c->has_vertex(v) );
  if ( cells.find( c ) != cells.end() )
    return; // c was already found

  cells.insert( c );
      
  for ( int j=0; j<4; j++ )
    if ( j != c->index(v) ) incident_cells( v, cells, c->neighbor(j) );
}


void Delaunay::incident_vertices(DVertex* v, std::set<DVertex*>& vertices,
		  DCell* c ) const
{
  assert( v != 0 );
  assert( d_mesh.is_vertex(v) );
      
  if ( number_of_vertices() < 2 ) return;

  if ( c == 0 ) c = v->cell();
  else assert( c->has_vertex(v) );

  std::set<DCell*> cells;
  util_incident_vertices(v, vertices, cells, c);
}

void Delaunay::util_incident_vertices(DVertex* v, 
		       std::set<DVertex*>& vertices,
		       std::set<DCell*>& cells, DCell* c ) const
{
  if ( cells.find( c ) != cells.end() )
    return; // c was already visited

  cells.insert( c );

  int d = dimension();
  for (int j=0; j <= d; j++ )
    if ( j != c->index(v) ) {
      if ( vertices.find( c->vertex(j) ) == vertices.end() )
	vertices.insert( c->vertex(j) );
      util_incident_vertices( v, vertices, cells, c->neighbor(j) );
    }
}

bool Delaunay::
is_valid(bool verbose, int level) const
{
  if ( ! mesh().is_valid(verbose,level) ) {
    if (verbose)
      std::cerr << "invalid data structure" << std::endl;
    assert(false);
    return false;
  }
    
  if ( &(*infinite_vertex()) == 0 ) {
    if (verbose)
	std::cerr << "no infinite vertex" << std::endl;
    assert(false);
    return false;
  }

  switch ( dimension() ) {
  case 3:
    {
      DCellIterator<VolumeMesh> it;
      for ( it = finite_cells_begin(); it != cells_end(); ++it ) {
	is_valid_finite(&(*it));
	for (int i=0; i<4; i++ ) {
	  if ( side_of_sphere ( &(*it), 
		 it->vertex( (it->neighbor(i))->index(&(*it)) )
		 ->point() ) == ON_BOUNDED_SIDE ) {
	    if (verbose)
	      std::cerr << "non-empty sphere " << std::endl;
	    assert(false);
	    return false;
	  }
	}
      }
      break;
    }
  case 2:
    {
      DFacetIterator<VolumeMesh> it;
      for ( it = finite_facets_begin(); it != facets_end(); ++it ) {
	is_valid_finite((*it).first);
	for (int i=0; i<2; i++ ) {
	  if ( side_of_circle ( (*it).first, 3,
		 (*it).first->vertex( (((*it).first)->neighbor(i))
			    ->index((*it).first) )->point() )
	       == ON_BOUNDED_SIDE ) {
	    if (verbose)
	      std::cerr << "non-empty circle " << std::endl;
	    assert(false);
	    return false;
	  }
	}
      }
      break;
    }
  case 1:
    {
      DEdgeIterator<VolumeMesh> it;
      for ( it = finite_edges_begin(); it != edges_end(); ++it )
	is_valid_finite((*it).first);
      break;
    }
  }
  if (verbose)
      std::cerr << "Delaunay valid triangulation" << std::endl;
  return true;
}



bool Delaunay::is_valid_finite(DCell* c, bool verbose, int) const
{
  switch ( dimension() ) {
  case 3:
    {
      if ( orientation(c->vertex(0)->point(),
	 	       c->vertex(1)->point(),
		       c->vertex(2)->point(),
		       c->vertex(3)->point()) != POSITIVE ) {
	if (verbose)
	  std::cerr << "badly oriented cell " 
		    << c->vertex(0)->point() << ", " 
		    << c->vertex(1)->point() << ", " 
		    << c->vertex(2)->point() << ", " 
		    << c->vertex(3)->point() << std::endl; 
 	assert(false);
	return false;
      }
      break;
    }
  case 2:
    {
      if (orientation(c->vertex(0)->point(),
		      c->vertex(1)->point(),
		      c->vertex(2)->point()) != POSITIVE) {
	if (verbose)
	  std::cerr << "badly oriented face "
		    << c->vertex(0)->point() << ", " 
		    << c->vertex(1)->point() << ", " 
		    << c->vertex(2)->point() << std::endl;
	assert(false);
	return false;
      }
      break;
    }
  case 1:
    {
      const Point & p0 = c->vertex(0)->point();
      const Point & p1 = c->vertex(1)->point();
	    
      if ( ! is_infinite ( c->neighbor(0)->vertex(c->neighbor(0)->index(c)) ) )
      {
 	const Point & n0 =
	  c->neighbor(0)->vertex(c->neighbor(0)->index(c))->point();  
 	if ( ( compare_x( p0, p1 ) != compare_x( p1, n0 ) )
	     || ( compare_y( p0, p1 ) != compare_y( p1, n0 ) )
	     || ( compare_z( p0, p1 ) != compare_z( p1, n0 ) ) ) {
	  if (verbose)
	    std::cerr << "badly oriented edge "
		      << p0 << ", " << p1 << std::endl
		      << "with neighbor 0"
		      << c->neighbor(0)->vertex(1-c->neighbor(0)->index(c))
	      ->point() << ", " << n0 << std::endl;
 	  assert(false);
	  return false;
 	}
      }
      if ( ! is_infinite ( c->neighbor(1)->vertex(c->neighbor(1)->index(c)) ) )
      {
	const Point & n1 = 
	  c->neighbor(1)->vertex(c->neighbor(1)->index(c))->point();
	if ( ( compare_x( p1, p0 ) != compare_x( p0, n1 ) )
	     || ( compare_y( p1, p0 ) != compare_y( p0, n1 ) )
	     || ( compare_z( p1, p0 ) != compare_z( p0, n1 ) ) ) {
	  if (verbose)
	    std::cerr << "badly oriented edge "
		      << p0 << ", " << p1 << std::endl
		      << "with neighbor 1"
		      << c->neighbor(1)->vertex(1-c->neighbor(1)->index(c))
	      ->point() << ", " << n1 << std::endl;
 	  assert(false);
	  return false;
 	}
      }
      break;
    }
  }
  return true;
}

  DVertex* Delaunay::insert(const Point &p, vector<triple<VMVertex*,VMVertex*,VMVertex*> >& facets, DCell* start, DVertex* v )
{
  Locate_type lt;
  int li, lj;
  DCell* c;

  c = locate( p, lt, li, lj, start);
  if ( lt == VERTEX ) return c->vertex(li);
  set_number_of_vertices(number_of_vertices()+1);
  Conflict_tester_3 tester(p, this);
  v = (DVertex*)d_mesh.insert_conflict(v, c, tester, facets); 
  v->set_point(p);
  return v;
}


DVertex* Delaunay::insert(const Point& p, DCell* start, DVertex* v)
{
  Locate_type lt;
  int li, lj;
  DCell* c;
  switch (dimension()) {
  case 3://dim 3
    {
    c = locate( p, lt, li, lj, start);
    if ( lt == VERTEX ) return c->vertex(li);
    set_number_of_vertices(number_of_vertices()+1);
    Conflict_tester_3 tester(p, this);
    v = (DVertex*)d_mesh.insert_conflict(v, c, tester); 
    v->set_point(p);
    return v;
    }
  case 2://dim2
    {
    c = locate( p, lt, li, lj, start);
    switch (lt) {
    case OUTSIDE_CONVEX_HULL:
    case CELL:
    case FACET:
    case EDGE:
      {
      set_number_of_vertices(number_of_vertices()+1);
      Conflict_tester_2 tester(p, this);
      v = (DVertex*) d_mesh.insert_conflict(v, c, tester); 
      v->set_point(p);
      return v;
      }
    case VERTEX:
      return c->vertex(li);
    case OUTSIDE_AFFINE_HULL:
      // if the 2d triangulation is Delaunay, the 3d
      // triangulation will be Delaunay
      return insert_outside_affine_hull(p,v); 
    }
    }
  default :
    // dimension <= 1
    return insert_in_element(p,start,v);//old triangulation insert
  }
}// insert(p)

Bounded_side
Delaunay::side_of_circle(DCell* c, int i, const Point & p) const
  // precondition : dimension >=2
  // in dimension 3, - for a finite facet
  // returns ON_BOUNDARY if the point lies on the circle,
  // ON_UNBOUNDED_SIDE when exterior, ON_BOUNDED_SIDE
  // interior
  // for an infinite facet, considers the plane defined by the
  // adjacent finite facet of the same cell, and does the same as in 
  // dimension 2 in this plane
  // in dimension 2, for an infinite facet
  // in this case, returns ON_BOUNDARY if the point lies on the 
  // finite edge (endpoints included) 
  // ON_BOUNDED_SIDE for a point in the open half-plane
  // ON_UNBOUNDED_SIDE elsewhere
{
  assert( dimension() >= 2 );
  int i3 = 5;

  if ( dimension() == 2 ) {
    assert( i == 3 );
    // the triangulation is supposed to be valid, ie the facet
    // with vertices 0 1 2 in this order is positively oriented
    if ( ! c->has_vertex( infinite_vertex(), i3 ) ) 
      return side_of_bounded_circle( c->vertex(0)->point(),
				     c->vertex(1)->point(),
				     c->vertex(2)->point(),
				     p );
    // else infinite facet
    // v1, v2 finite vertices of the facet such that v1,v2,infinite
    // is positively oriented
    DVertex* v1 = c->vertex( ccw(i3) );
    DVertex* v2 = c->vertex( cw(i3) );
    assert(orientation(v1->point(), v2->point(),
		       (c->mirror_vertex(i3))->point()) == NEGATIVE);
    Orientation o = orientation(v1->point(), v2->point(), p);
    if ( o != ZERO ) return Bounded_side( o );
    // because p is in f iff
    // is does not lie on the same side of v1v2 as vn
    int i_e;
    Locate_type lt;
    // case when p collinear with v1v2
    return side_of_segment( p, v1->point(), v2->point(), lt, i_e );
  }

  // else dimension == 3
  assert( i >= 0 && i < 4 );
  if ( ( ! c->has_vertex(infinite_vertex(),i3) ) || ( i3 != i ) ) {
    // finite facet
    // initialization of i0 i1 i2, vertices of the facet positively 
    // oriented (if the triangulation is valid)
    int i0 = (i>0) ? 0 : 1;
    int i1 = (i>1) ? 1 : 2;
    int i2 = (i>2) ? 2 : 3;
    assert( orientation( c->vertex(i0)->point(),
			 c->vertex(i1)->point(),
			 c->vertex(i2)->point(),
			 p ) == COPLANAR );
    return side_of_bounded_circle( c->vertex(i0)->point(),
				   c->vertex(i1)->point(),
				   c->vertex(i2)->point(),
				   p );
  }

  //else infinite facet
  // v1, v2 finite vertices of the facet such that v1,v2,infinite
  // is positively oriented
  DVertex* v1 = c->vertex( next_around_edge(i3,i) );
  DVertex* v2 = c->vertex( next_around_edge(i,i3) );
  Orientation o = (Orientation)
    (orientation( v1->point(), v2->point(),
		  c->vertex(i)->point()) *
     orientation( v1->point(), v2->point(), p ));
  // then the code is duplicated from 2d case
  if ( o != ZERO ) return Bounded_side( -o );
  // because p is in f iff 
  // it is not on the same side of v1v2 as c->vertex(i)
  int i_e;
  Locate_type lt;
  // case when p collinear with v1v2
  return side_of_segment( p, v1->point(), v2->point(), lt, i_e );
}

Bounded_side
Delaunay::side_of_sphere(DCell* c, const Point & p) const
{
  assert( dimension() == 3 );
  int i3;
  if ( ! c->has_vertex( infinite_vertex(), i3 ) ) 
    return Bounded_side( side_of_oriented_sphere( c->vertex(0)->point(),
						  c->vertex(1)->point(),
						  c->vertex(2)->point(),
						  c->vertex(3)->point(),
						  p ) );
  // else infinite cell :
  unsigned char i[3] = {(i3+1)&3, (i3+2)&3, (i3+3)&3};
  if ( (i3&1) == 0 )
      std::swap(i[0], i[1]);
  Orientation o = orientation( c->vertex(i[0])->point(),
			       c->vertex(i[1])->point(),
			       c->vertex(i[2])->point(),
			       p );
  if (o != ZERO)
    return Bounded_side(o);

  return side_of_bounded_circle( c->vertex(i[0])->point(), 
				 c->vertex(i[1])->point(),
				 c->vertex(i[2])->point(),
				 p );
}

bool Delaunay::is_edge(DVertex* v0, DVertex* v1)
{
  set<DVertex*> vertices;
  incident_vertices(v0,vertices);
  if ( vertices.find(v1) != vertices.end() ) return true;
  return false;
}

bool Delaunay::is_face(DVertex* v0, DVertex* v1, DVertex* v2)
{
  set<DCell*> cells;
  set<DCell*>::const_iterator it;
  DCell *cell;
  DVertex *vt0, *vt1, *vt2, *vt3;
  incident_cells(v0,cells);
  for ( it = cells.begin(); it != cells.end(); it++ ) {
    cell = *it;
    vt0 = cell->vertex(0);
    vt1 = cell->vertex(1);
    vt2 = cell->vertex(2);
    vt3 = cell->vertex(3);
    if ( v0 != vt0 && v0 != vt1 && v0 != vt2 && v0 != vt3 ) continue;
    if ( v1 != vt0 && v1 != vt1 && v1 != vt2 && v1 != vt3 ) continue;
    if ( v2 != vt0 && v2 != vt1 && v2 != vt2 && v2 != vt3 ) continue;
    return true;
  }
  return false;
}

void Delaunay::label_cells(hash_map<Face,BFace*,FaceHash,FaceEqual>& bfaces)
{
  DCell *c, *nc;
  stack<DCell*> clist;
  DVertex *v1, *v2, *v3;
  int i;
  DCellIterator<VolumeMesh> ci = all_cells_begin();
  while ( ci != cells_end() ) {
    (*ci).set_domain(1);
    ci++;
  }
  c = infinite_vertex()->cell();
  c->set_domain(0);
  clist.push(c);
  while ( !clist.empty() ) {
    c = clist.top();
    clist.pop();
    for ( i = 0; i < 4; i++ ) {
      nc = c->neighbor(i);
      if ( nc->is_test() ) continue;
      v1 = c->vertex((i+1)&3);
      v2 = c->vertex((i+2)&3);
      v3 = c->vertex((i+3)&3);
      if ( bfaces.find(make_triple(v1,v2,v3)) == bfaces.end() ) {
	nc->set_domain(0);
	clist.push(nc);
      }
    }
  }
}

}

#endif

