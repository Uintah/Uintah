#ifndef SCI_Wangxl_Datatypes_Mesh_VolumeMesh_h
#define SCI_Wangxl_Datatypes_Mesh_VolumeMesh_h

#include <utility>
#include <map>
#include <set>
#include <vector>

#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/Utilities.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertexBase.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCellBase.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCell.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMVertex.h>

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMIterators.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCirculators.h>

namespace Wangxl {

using namespace SCIRun;

class VolumeMesh : public Utilities
{
public:
  friend class VMCellIterator<VolumeMesh>;
  friend class VMFacetIterator<VolumeMesh>;
  friend class VMEdgeIterator<VolumeMesh>;
  friend class VMVertexIterator<VolumeMesh>;

  friend class VMCellCirculator<VolumeMesh>;
  friend class VMFacetCirculator<VolumeMesh>;
  VolumeMesh() : d_dimension(-2), d_number_of_vertices(0)
  {
      init_cell_list(&d_list_of_cells);
      init_cell_list(&d_list_of_free_cells);
      init_cell_list(&d_list_of_temporary_free_cells);
  }

  VolumeMesh(const  VolumeMesh & mesh) : d_number_of_vertices(0)
    // _number_of_vertices is set to 0 so that clear() in copy_tds() works
  {
    init_cell_list(&d_list_of_cells);
    init_cell_list(&d_list_of_free_cells);
    init_cell_list(&d_list_of_temporary_free_cells);
    copy_mesh(mesh);
  }

  ~VolumeMesh()
  {
    clear();
  }

  VolumeMesh & operator= (const  VolumeMesh & mesh)
  {
    copy_mesh(mesh);
    return *this;
  }  

  int number_of_vertices() const {return d_number_of_vertices;}
  
  int dimension() const {return d_dimension;}

  int number_of_cells() const 
    { 
      if ( dimension() < 3 ) return 0;
      return std::distance(cells_begin(), cells_end());
    }
  
  int number_of_facets() const
    {
      if ( dimension() < 2 ) return 0;
      return std::distance(facets_begin(), facets_end());
    }

  int number_of_edges() const
    {
      if ( dimension() < 1 ) return 0;
      return std::distance(edges_begin(), edges_end());
    }

  // USEFUL CONSTANT TIME FUNCTIONS

  // SETTING
  // to be protected ?

  void set_number_of_vertices(int n) { d_number_of_vertices = n; }

  void set_dimension(int n) { d_dimension = n; }

  VMVertex* create_vertex()
  {
      return new VMVertex();
  }

  VMCell* create_cell() 
    { 
      VMCell* c = get_new_cell();
      put_cell_in_list(c, d_list_of_cells);
      return c; 
    }

  VMCell* create_cell(VMCell* c)
    {
      VMCell* cnew = get_new_cell();
      *cnew = c;
      cnew->init();
      put_cell_in_list(cnew, d_list_of_cells);
      return cnew; 
    }

  VMCell* create_cell(VMVertex* v0, VMVertex* v1, VMVertex* v2, VMVertex* v3)
    {
      VMCell* c = get_new_cell();
      c->set_vertices(v0,v1,v2,v3);
      put_cell_in_list(c, d_list_of_cells);
      return c; 
    }

  VMCell* create_cell(VMVertex* v0, VMVertex* v1, VMVertex* v2, VMVertex* v3,
		    VMCell* n0, VMCell* n1, VMCell* n2, VMCell* n3)
    {
      VMCell* c = get_new_cell();
      c->set_vertices(v0,v1,v2,v3);
      c->set_neighbors(n0,n1,n2,n3);
      put_cell_in_list(c, d_list_of_cells);
      return c; 
    }

private:

  // Used to initialize the lists to empty lists.
  void init_cell_list(VMCell* c)
  {
      c->d_next = c;
      c->d_previous = c;
  }

  void link_cells(VMCell* a, VMCell *b)
  {
      a->d_next = b;
      b->d_previous = a;
  }

  void remove_cell_from_list(VMCell* c)
  {
      link_cells(c->d_previous, c->d_next);
  }

  VMCell* get_new_cell()
  {
      VMCell *r;
      if (d_list_of_free_cells.d_next == &d_list_of_free_cells)
      {
	  // We create a new array.
	  //cell_array_vector.push_back(new Cell[1000]);
	  //for (int i=0; i<1000; ++i)

	  r = new VMCell();
      }
      else
      {
          r = d_list_of_free_cells.d_next;
          r->set_conflict_flag(0);
	  r->init();
          remove_cell_from_list(r);
      }
      r->init();
      return r;
  }

  void move_cell_to_temporary_free_list(VMCell *c)
  {
      remove_cell_from_list(c);
      put_cell_in_list(c, d_list_of_temporary_free_cells);
  }

  void move_temporary_free_cells_to_free_list()
  {
      assert( &d_list_of_temporary_free_cells !=
	     d_list_of_temporary_free_cells.d_next );

      link_cells(d_list_of_temporary_free_cells.d_previous,
	         d_list_of_free_cells.d_next);
      link_cells(&d_list_of_free_cells,
	         d_list_of_temporary_free_cells.d_next);
      link_cells(&d_list_of_temporary_free_cells,
	         &d_list_of_temporary_free_cells);
  }

  void put_cell_in_list(VMCell *c, VMCell &l)
  {
      assert( c != 0 );
      link_cells(c, l.d_next);
      link_cells(&l, c);
  }

public:
  // not documented
  void read_cells(std::istream& is, std::map< int, VMVertex* > &V,
			   int & m, std::map< int, VMCell* > &C );
  // ACCESS FUNCTIONS

  void delete_vertex( VMVertex* v )
  {
      // We can't check this condition because vertices are not linked in a
      // list, unlike the cells.
      // CGAL_triangulation_expensive_precondition( is_vertex(v) );
      delete v;
  }

  void delete_cell( VMCell* c )
    { 
      assert( dimension() != 3 || is_cell(c) );
      assert( dimension() != 2 || is_facet(c,3) );
      assert( dimension() != 1 || is_edge(c,0,1) );
      assert( dimension() != 0 || is_vertex(c->vertex(0)) );

      remove_cell_from_list(c);
      put_cell_in_list(c, d_list_of_free_cells);
      // Maybe we should have an heuristic to know when to really
      // delete the cells, or provide some flush() method to the user.
    }

  // QUERIES

  bool is_vertex(VMVertex* v) const;
  bool is_edge(VMCell* c, int i, int j) const;
  bool is_edge(VMVertex* u, VMVertex* v, VMCell* & c, int & i, int & j) const;
  bool is_facet(VMCell* c, int i) const;
  bool is_facet(VMVertex* u, VMVertex* v, VMVertex* w, 
		VMCell* & c, int & i, int & j, int & k) const;
  bool is_cell(VMCell* c) const;
  bool is_cell(VMVertex* u, VMVertex* v, VMVertex* w, VMVertex* t, 
	       VMCell* & c, int & i, int & j, int & k, int & l) const;
  bool is_cell(VMVertex* u, VMVertex* v, VMVertex* w, VMVertex* t) const; 

  bool has_vertex(const VMFacet & f, VMVertex* v, int & j) const;
  bool has_vertex(VMCell* c, int i, VMVertex* v, int & j) const;
  bool has_vertex(const VMFacet & f, VMVertex* v) const;
  bool has_vertex(VMCell* c, int i, VMVertex* v) const;

  bool are_equal(VMCell* c, int i, VMCell* n, int j) const;
  bool are_equal(const VMFacet & f, const VMFacet & g) const;
  bool are_equal(const VMFacet & f, VMCell* n, int j) const;

public:

  //INSERTION

  VMVertex * insert_in_cell(VMVertex * v, VMCell* c);

  VMVertex * insert_in_facet(VMVertex * v, const VMFacet & f)
    { return insert_in_facet(v/*w*/,f.first,f.second); }
  
  VMVertex * insert_in_facet(VMVertex * v, VMCell* c, int i);
  
  VMVertex * insert_in_edge(VMVertex * v, const VMEdge & e)   
    { return insert_in_edge(v/*w*/, e.first, e.second, e.third); }
  
  VMVertex * insert_in_edge(VMVertex * v, VMCell* c, int i, int j);   

  VMVertex * insert_increase_dimension(VMVertex * v, // new vertex
				     VMVertex* star = 0,
				     bool reorient = false);

private:

  template <class Conflict_test, /*class it_boundary_facets,
				   class it_cells,*/ class it_internal_facets>
  void
  find_conflicts_3(VMCell* c, VMCell* &ac, const Conflict_test &tester,
                   /*it_boundary_facets fit, it_cells cit,*/
                   it_internal_facets ifit) const
  {
    assert( dimension()==3 );
    assert( tester(c) );

    c->set_conflict_flag(1);
    /* *cit++ = c; */

    for (int i=0; i<4; ++i) {
      VMCell* test = c->neighbor(i);
      if (test->get_conflict_flag() == 1) { // test was already in conflict.
          if (c < test)
	    *ifit++ = make_triple(c->vertex((i+1)&3),c->vertex((i+2)&3),c->vertex((i+3)&3)); // Internal facet.
          continue;
      }
      if (test->get_conflict_flag() == 0) {
          if (tester(test)) {
              if (c < test)
                  *ifit++ = make_triple(c->vertex((i+1)&3),c->vertex((i+2)&3),c->vertex((i+3)&3)); // Internal facet.
              find_conflicts_3(test, ac, tester, /*fit, cit, */ifit);
              continue;
          }
	  test->set_conflict_flag(/*2*/-1); // test is on the boundary.
	  ac = c;
      }
      /* *fit++ = Facet(c, i);*/
    }
  }



  template < class Conflict_test >
  void
  find_conflicts_3(VMCell* c, VMCell* &ac, int &i, const Conflict_test &tester,vector<VMCell*>& ccells)
  {
    // The semantic of the flag is the following :
    // 0  -> never went on the cell
    // 1  -> cell is in conflict
    // -1 -> cell is not in conflict

    assert( tester(c) );
    ccells.push_back(c);
    //    cout << " ccells size: " << ccells.size() << endl;
    move_cell_to_temporary_free_list(c);
    c->set_conflict_flag(1);

    for ( int j=0; j<4; j++ ) {
      VMCell* test = c->neighbor(j);
      if (test->get_conflict_flag() != 0)
        continue; // test was already tested.
      if ( tester(test) )
        find_conflicts_3(test, ac, i, tester, ccells);
      else {
        test->set_conflict_flag(-1);
        ac = c;
        i = j;
      }
    }
  }
  

  // The two find_conflicts_[23] below could probably be merged ?
  // The only difference between them is the test "j<3" instead of "j<4"...
  template < class Conflict_test >
  void
  find_conflicts_3(VMCell* c, VMCell* &ac, int &i, const Conflict_test &tester)
  {
    // The semantic of the flag is the following :
    // 0  -> never went on the cell
    // 1  -> cell is in conflict
    // -1 -> cell is not in conflict

    assert( tester(c) );

    move_cell_to_temporary_free_list(c);
    c->set_conflict_flag(1);

    for ( int j=0; j<4; j++ ) {
      VMCell* test = c->neighbor(j);
      if (test->get_conflict_flag() != 0)
        continue; // test was already tested.
      if ( tester(test) )
        find_conflicts_3(test, ac, i, tester);
      else {
        test->set_conflict_flag(-1);
        ac = c;
        i = j;
      }
    }
  }

  template < class Conflict_test >
  void
  find_conflicts_2(VMCell *c, VMCell * &ac, int &i, const Conflict_test &tester)
  {
    assert( tester(c) );

    move_cell_to_temporary_free_list(c);
    c->set_conflict_flag(1);

    for ( int j=0; j<3; j++ ) {
      VMCell * test = c->neighbor(j);
      if (test->get_conflict_flag() != 0)
        continue; // test was already tested.
      if ( tester(test) )
        find_conflicts_2(test, ac, i, tester);
      else {
        test->set_conflict_flag(-1);
        ac = c;
        i = j;
      }
    }
  }

  VMCell * create_star_3(VMVertex* v, VMCell* c, int li,
	               VMCell * prev_c = 0, VMVertex * prev_v = 0);
  VMCell * create_star_2(VMVertex* v, VMCell* c, int li );

public:


  template < class Conflict_test >
  VMVertex * insert_conflict( VMVertex * w, VMCell *c, const Conflict_test &tester, vector<triple<VMVertex*,VMVertex*,VMVertex*> >& facets)
{
  assert( dimension() >= 2 );
  assert( c != 0 );
  assert( tester(c) );
  
  if ( w == 0 ) 
    w = create_vertex();
  // Find the cells in conflict.
  VMCell *ccc;
  int i,ii;


  find_conflicts_3(c, ccc, tester,std::back_inserter(facets));

  // Create the new cells, and returns one of them.
  VMCell * nouv = create_star_3( w, ccc, i );
  w->set_cell( nouv );
  
  move_temporary_free_cells_to_free_list();
  return w;
}

  // This one takes a function object to recursively determine the cells in
  // conflict, then inserts by starring.
  // Maybe we need _2 and _3 versions ?
  template < class Conflict_test >
  VMVertex * insert_conflict( VMVertex * w, VMCell *c, const Conflict_test &tester)
  {
    assert( dimension() >= 2 );
    assert( c != 0 );
    assert( tester(c) );

    if ( w == 0 ) 
      w = create_vertex();

    if (dimension() == 3)
    {
      // Find the cells in conflict.
      VMCell *ccc;
      int i;

      find_conflicts_3(c, ccc, i, tester);

      // Create the new cells, and returns one of them.
      VMCell * nouv = create_star_3( w, ccc, i );
      w->set_cell( nouv );

      move_temporary_free_cells_to_free_list();
    }
    else // dim == 2
    {
      // Find the cells in conflict.
      VMCell *ccc;
      int i;
      find_conflicts_2(c, ccc, i, tester);

      // Create the new cells, and returns one of them.
      VMCell * nouv = create_star_2( w, ccc, i );
      w->set_cell( nouv );

      move_temporary_free_cells_to_free_list();
    }

    return w;
  }

  // ITERATOR METHODS

  VMCellIterator<VolumeMesh> cells_begin() const
  {
    if ( dimension() < 3 )
	return cells_end();
    return VMCellIterator<VolumeMesh>(this);
  }

  VMCellIterator<VolumeMesh> cells_end() const
  {
    return VMCellIterator<VolumeMesh>(this, 1);
  }

  VMFacetIterator<VolumeMesh> facets_begin() const
  {
    if ( dimension() < 2 )
	return facets_end();
    return VMFacetIterator<VolumeMesh>(this);
  }

  VMFacetIterator<VolumeMesh> facets_end() const
  {
    return VMFacetIterator<VolumeMesh>(this, 1);
  }

  VMEdgeIterator<VolumeMesh> edges_begin() const
  {
    if ( dimension() < 1 )
	return edges_end();
    return VMEdgeIterator<VolumeMesh>(this);
  }

  VMEdgeIterator<VolumeMesh> edges_end() const
  {
    return VMEdgeIterator<VolumeMesh>(this,1);
  }

  VMVertexIterator<VolumeMesh> vertices_begin() const
  {
    if ( number_of_vertices() <= 0 )
	return vertices_end();
    return VMVertexIterator<VolumeMesh>(this);
  }

  VMVertexIterator<VolumeMesh> vertices_end() const
  {
    return VMVertexIterator<VolumeMesh>(this, 1);
  }

  // CIRCULATOR METHODS

  // cells around an edge
  VMCellCirculator<VolumeMesh> incident_cells(const VMEdge & e) const
    {
      assert( dimension() == 3 );
      return VMCellCirculator<VolumeMesh>(this, e);
    }
 VMCellCirculator<VolumeMesh> incident_cells(VMCell* ce, int i, int j) const
  {
    assert( dimension() == 3 );
    return VMCellCirculator<VolumeMesh>(this, ce, i, j);
  }

 VMCellCirculator<VolumeMesh> incident_cells(const VMEdge & e, VMCell* start) const
  {
    assert( dimension() == 3 );
    return VMCellCirculator<VolumeMesh>(this, e, start);
  }
 VMCellCirculator<VolumeMesh> incident_cells(VMCell* ce, int i, int j, VMCell* start) const
  {
    assert( dimension() == 3 );
    return VMCellCirculator<VolumeMesh>(this, ce, i, j, start);
  }

  //facets around an edge
  VMFacetCirculator<VolumeMesh> incident_facets(const VMEdge & e) const
  {
    assert( dimension() == 3 );
    return VMFacetCirculator<VolumeMesh>(this, e);
  }
  VMFacetCirculator<VolumeMesh> incident_facets(VMCell* ce, int i, int j) const
  {
    assert( dimension() == 3 );
    return VMFacetCirculator<VolumeMesh>(this, ce, i, j);
  }
  VMFacetCirculator<VolumeMesh> incident_facets(const VMEdge & e, const VMFacet & start) const
  {
    assert( dimension() == 3 );
    return VMFacetCirculator<VolumeMesh>(this, e, start);
  }
  VMFacetCirculator<VolumeMesh> incident_facets(VMCell* ce, int i, int j,
				   const VMFacet & start) const
  {
    assert( dimension() == 3 );
    return VMFacetCirculator<VolumeMesh>(this, ce, i, j, start);
  }
  VMFacetCirculator<VolumeMesh> incident_facets(const VMEdge & e, VMCell* start, int f) const
  {
    assert( dimension() == 3 );
    return VMFacetCirculator<VolumeMesh>(this, e, start, f);
  }
  VMFacetCirculator<VolumeMesh> incident_facets(VMCell* ce, int i, int j, 
				  VMCell* start, int f) const
  {
    assert( dimension() == 3 );
    return VMFacetCirculator<VolumeMesh>(this, ce, i, j, start, f);
  }

  // around a vertex
  void
  incident_cells(VMVertex* v, std::set<VMCell*> & cells, VMCell* c = 0 ) const;

  void
  incident_vertices(VMVertex* v, std::set<VMVertex*> & vertices,
		    VMCell* c = 0 ) const;

  // CHECKING
  bool is_valid(bool verbose = false, int level = 0) const;


  // Helping functions
  void init(VMVertex* v) {} // What's the purpose ???

  VMVertex* copy_mesh(const VolumeMesh & mesh, VMVertex* vert = 0);
    // returns the new vertex corresponding to vert in the new tds 

  void clear();

  void clear_cells_only(std::vector<VMVertex *> & Vertices);


private:
  // in dimension i, number of vertices >= i+2 
  // ( the boundary of a simplex in dimension i+1 has i+2 vertices )
  int d_dimension;
  int d_number_of_vertices;
  
  // we maintain the list of cells to be able to traverse the triangulation
  // it starts with a "foo" element that will never be removed.
  // the list is circular, the foo element being used to recognize the end
  // of the list
  VMCell d_list_of_cells;

  // The 2 following free cells lists do not need to be doubly connected.
  // We could then use the second pointer to store the flag, then ?
  //         _previous_cell == 0 for when in conflict
  //         _previous_cell |= 1 for when not in conflict ?
  // This is a list of free cells that serves as an allocator cache.
  VMCell d_list_of_free_cells;

  // This is a list of cells that is filled by find_conflicts, and which is
  // merged to _list_of_free_cells after create_star.
  VMCell d_list_of_temporary_free_cells;

  // Cells and vertices allocation by arrays.
  //std::vector<Cell[1000] *> cell_array_vector;
  //std::vector<Vertex[1000] *> vertex_array_vector;

  // ACCESS FUNCTIONS

  VMCell & list_of_cells() 
    {return d_list_of_cells;}
  
  VMCell* past_end_cell() const 
    {
      return &( const_cast<VolumeMesh *>(this)->d_list_of_cells );
    } 

  // used by is-valid :
  bool count_vertices(int & i, bool verbose = false, int level = 0) const;
  // counts AND checks the validity
  bool count_facets(int & i, bool verbose = false, int level = 0) const;
  // counts but does not check
  bool count_edges(int & i, bool verbose = false, int level = 0) const;
  // counts but does not check
  bool count_cells(int & i, bool verbose = false, int level = 0) const;
  // counts AND checks the validity
};


bool VolumeMesh::is_vertex(VMVertex* v) const
{
  VMVertexIterator<VolumeMesh> it = vertices_begin();
  while (it != vertices_end()) {
    if ( v == &(*it) )
      return true;
    ++it;
  }
  return false;
}

bool VolumeMesh::is_edge(VMVertex* u, VMVertex* v, VMCell* & c, int & i, int & j) const
  // returns false when dimension <1 or when indices wrong
{
  if (u==v)
      return false;
  
  VMCell* tmp = d_list_of_cells.d_next;
  while ( tmp != past_end_cell() ) {
    if ( (tmp->has_vertex(u,i)) && (tmp->has_vertex(v,j)) ) {
      c = tmp;
      return true; 
    }
    tmp = tmp->d_next;
  }
  return false;
} 

bool VolumeMesh::is_edge(VMCell* c, int i, int j) const
  // returns false when dimension <1
{
  if ( i==j ) return false;
  if ( (i<0) || (j<0) ) return false;
  if ( (dimension() == 1) && ((i>1) || (j>1)) ) return false;
  if ( (dimension() == 2) && ((i>2) || (j>2)) ) return false;
  if ((i>3) || (j>3)) return false;

  VMCell* tmp = d_list_of_cells.d_next;
  while ( tmp != past_end_cell() ) {
    if (tmp == c) return true;
    tmp = tmp->d_next;
  }
  return false;
}

bool VolumeMesh::is_facet(VMVertex* u, VMVertex* v, VMVertex* w, 
	 VMCell* & c, int & i, int & j, int & k) const
  // returns false when dimension <2 or when indices wrong
{
  if ( (u==v) || (u==w) || (v==w) ) return false;
  VMFacetIterator<VolumeMesh> it = facets_begin();
  while ( it != facets_end() ) {
    if ( ( ((*it).first)->has_vertex(u,i) )
	 && ( ((*it).first)->has_vertex(v,j) )
	 && ( ((*it).first)->has_vertex(w,k) ) ) {
      c = (*it).first;
      return true;
    }
    ++it;
  }
  return false;
}

bool VolumeMesh::is_facet(VMCell* c, int i) const
  // returns false when dimension <2
{
  if (i<0) return false;
  if ( (dimension() == 2) && (i!=3) ) return false;
  if (i>3) return false;
  VMFacetIterator<VolumeMesh> it = facets_begin();
  while ( it != facets_end() ) {
    if ( (*it).first == c ) return true;
    ++it;
  }
  return false;
}

bool VolumeMesh::is_cell( VMCell* c ) const
  // returns false when dimension <3
{
  if ( c == 0 ) return false;
  VMCellIterator<VolumeMesh> it = cells_begin();
  while ( it != cells_end() ) {
    if ( c == &(*it) ) {
      return true;
    }
    ++it;
  }
  return false;
}

bool VolumeMesh::is_cell(VMVertex* u, VMVertex* v, VMVertex* w, VMVertex* t, 
			 VMCell* & c, int & i, int & j, int & k, int & l) const
  // returns false when dimension <3
{
  if ( (u==v) || (u==w) || (u==t) || (v==w) || (v==t) || (w==t) )
    return false;
  VMCellIterator<VolumeMesh> it = cells_begin();
  while ( it != cells_end() ) {
    if ( ( it->has_vertex(u,i) )
	 && ( it->has_vertex(v,j) )
	 && ( it->has_vertex(w,k) ) 
	 && ( it->has_vertex(t,l) ) ) {
      c = &(*it);
      return true;
    }
    ++it;
  }
  return false;
}

bool VolumeMesh::is_cell(VMVertex* u, VMVertex* v, VMVertex* w, VMVertex* t) const
  // returns false when dimension <3
{
  if ( (u==v) || (u==w) || (u==t) || (v==w) || (v==t) || (w==t) )
    return false;
  VMCellIterator<VolumeMesh> it = cells_begin();
  while ( it != cells_end() ) {
    if ( ( it->has_vertex(u) ) &&
	 ( it->has_vertex(v) ) &&
	 ( it->has_vertex(w) ) &&
	 ( it->has_vertex(t) ) ) {
      return true;
    }
    ++it;
  }
  return false;
}

bool VolumeMesh::has_vertex(VMCell* c, int i, VMVertex* v, int & j) const
  // computes the index j of the vertex in the cell c giving the query
  // facet (c,i)  
  // j has no meaning if false is returned
{
  assert( dimension() == 3 ); 
  return ( c->has_vertex(v,j) && (j != i) );
}

bool VolumeMesh::has_vertex(VMCell* c, int i, VMVertex* v) const
  // checks whether the query facet (c,i) has vertex v
{
  assert( dimension() == 3 ); 
  int j;
  return ( c->has_vertex(v,j) && (j != i) );
}

bool VolumeMesh::has_vertex(const VMFacet & f, VMVertex* v, int & j) const
{
  return( has_vertex( f.first, f.second, v, j ) );
}

bool VolumeMesh::has_vertex(const VMFacet & f, VMVertex* v) const
{
  return( has_vertex( f.first, f.second, v ) );
}

bool VolumeMesh::are_equal(VMCell* c, int i, VMCell* n, int j) const
  // tests whether facets c,i and n,j, have the same 3 vertices
  // the triangulation is supposed to be valid, the orientation of the 
  // facets is not checked here
  // the neighbor relations between c and  n are not tested either,
  // which allows to use this method before setting these relations
  // (see remove in Delaunay_3)
  //   if ( c->neighbor(i) != n ) return false;
  //   if ( n->neighbor(j) != c ) return false;

{
  assert( dimension() == 3 ); 

  if ( (c==n) && (i==j) ) return true;

  int j1,j2,j3;
  return( n->has_vertex( c->vertex((i+1)&3), j1 ) &&
	  n->has_vertex( c->vertex((i+2)&3), j2 ) &&
	  n->has_vertex( c->vertex((i+3)&3), j3 ) &&
	  ( j1+j2+j3+j == 6 ) );
}

bool VolumeMesh::are_equal(const VMFacet & f, const VMFacet & g) const
{
  return( are_equal( f.first, f.second, g.first, g.second ) );
}

bool VolumeMesh::are_equal(const VMFacet & f, VMCell* n, int j) const
{
  return( are_equal( f.first, f.second, n, j ) );
}

VMVertex* VolumeMesh::insert_in_cell( VMVertex * v, VMCell* c )
{
  assert( dimension() == 3 );
  assert( (c != 0) );
  assert( is_cell(c) );

  if ( v == 0 )
    v = create_vertex();

  VMVertex* v0 = c->vertex(0);
  VMVertex* v1 = c->vertex(1);
  VMVertex* v2 = c->vertex(2);
  VMVertex* v3 = c->vertex(3);

  VMCell* n1 = c->neighbor(1);
  VMCell* n2 = c->neighbor(2);
  VMCell* n3 = c->neighbor(3);

  // c will be modified to have v,v1,v2,v3 as vertices
  VMCell* c3 = create_cell(v0,v1,v2,v,c,0,0,n3);
  VMCell* c2 = create_cell(v0,v1,v,v3,c,0,n2,c3);
  VMCell* c1 = create_cell(v0,v,v2,v3,c,n1,c2,c3);

  c3->set_neighbor(1,c1);
  c3->set_neighbor(2,c2);
  c2->set_neighbor(1,c1);

  n1->set_neighbor(n1->index(c),c1);
  n2->set_neighbor(n2->index(c),c2);
  n3->set_neighbor(n3->index(c),c3);

  c->set_vertex(0,v);
  c->set_neighbor(1,c1);
  c->set_neighbor(2,c2);
  c->set_neighbor(3,c3);

  if( v0->cell() == c  ) {  v0->set_cell(c1); }
  v->set_cell(c);
  set_number_of_vertices(number_of_vertices() +1);

  return v;
}

VMVertex* VolumeMesh::insert_in_facet(VMVertex * v, VMCell* c, int i)
{ // inserts v in the facet opposite to vertex i of cell c

  assert( (c != 0)); 
  assert( dimension() >= 2 );

  if ( v == 0 )
    v = create_vertex();

  switch ( dimension() ) {

  case 3:
    {
      assert( is_cell(c) );
      assert( i == 0 || i == 1 || 
				       i == 2 || i == 3 );
      // c will be modified to have v replacing vertex(i+3)
      int i1,i2,i3;

      if ( (i&1) == 0 ) {
	i1=(i+1)&3; i2=(i+2)&3; i3=6-i-i1-i2;
      }
      else {
	i1=(i+1)&3; i2=(i+3)&3; i3=6-i-i1-i2;
      }
      // i,i1,i2,i3 is well oriented
      // so v will "replace" the vertices in this order
      // when creating the new cells one after another from c

      VMVertex* vi=c->vertex(i);
      VMVertex* v1=c->vertex(i1); 
      VMVertex* v2=c->vertex(i2);
      VMVertex* v3=c->vertex(i3);

      // new cell with v in place of i1
      VMCell* nc = c->neighbor(i1);
      VMCell* cnew1 = create_cell(vi,v,v2,v3,
				0,nc,0,c);
      nc->set_neighbor(nc->index(c),cnew1);
      c->set_neighbor(i1,cnew1);

      v3->set_cell(cnew1);

      // new cell with v in place of i2
      nc = c->neighbor(i2);
      VMCell* cnew2 = create_cell(vi,v1,v,v3,
				0,cnew1,nc,c);
      nc->set_neighbor(nc->index(c),cnew2);
      c->set_neighbor(i2,cnew2);
      cnew1->set_neighbor(2,cnew2); // links to previous cell

      // v replaces i3 in c
      c->set_vertex(i3,v);

      // other side of facet containing v
      VMCell* d = c->neighbor(i);
      int j = d->index(c);
      int j1=d->index(v1);// triangulation supposed to be valid
      int j2=d->index(v2);
      int j3=6-j-j1-j2;
      // then the orientation of j,j1,j2,j3 depends on the parity
      // of i-j

      // new cell with v in place of j1
      VMCell* nd = d->neighbor(j1);
      VMCell* dnew1 = create_cell(d->vertex(j),v,v3,v2,
				cnew1,nd,d,0);
      nd->set_neighbor(nd->index(d),dnew1);
      d->set_neighbor(j1,dnew1);
      cnew1->set_neighbor(0,dnew1);
	  
      // new cell with v in place of j2
      nd = d->neighbor(j2);
      VMCell* dnew2 = create_cell(d->vertex(j),v1,v3,v,
				cnew2,dnew1,d,nd);
      nd->set_neighbor(nd->index(d),dnew2);
      d->set_neighbor(j2,dnew2);
      cnew2->set_neighbor(0,dnew2);
      dnew1->set_neighbor(3,dnew2);

      // v replaces i3 in d
      d->set_vertex(j3,v);
      v->set_cell(d);

      break;
    }
  case 2:
    {
      assert( is_facet(c,i) );
      VMCell* n = c->neighbor(2);
      VMCell* cnew = create_cell(c->vertex(0),c->vertex(1),v,0,
			       c, 0,n,0);
      n->set_neighbor(n->index(c),cnew);
      c->set_neighbor(2,cnew);
      c->vertex(0)->set_cell(cnew);

      n = c->neighbor(1);
      VMCell* dnew = create_cell(c->vertex(0),v,c->vertex(2),0,
			       c,n,cnew,0);
      n->set_neighbor(n->index(c),dnew);
      c->set_neighbor(1,dnew);
      cnew->set_neighbor(1,dnew);

      c->set_vertex(0,v);
      v->set_cell(c);
      break;
    }
  }
  set_number_of_vertices(number_of_vertices() +1);

  return v;
}
// end insert_in_facet

VMVertex* VolumeMesh::insert_in_edge(VMVertex * v, VMCell* c, int i, int j)   
  // inserts v in the edge of cell c with vertices i and j
{ 
  assert( c != 0 ); 
  assert( i != j );
  assert( dimension() >= 1 );

  if ( v == 0 )
    v = create_vertex();

  VMCell* cnew;
  VMCell* dnew;

  switch ( dimension() ) {
    
  case 3:
    {
      assert( is_cell(c) );
      assert( i>=0 && i<=3 && j>=0 && j<=3 );
      VMVertex* vi=c->vertex(i);
      VMVertex* vj=c->vertex(j);
	
      cnew = create_cell(c);
      c->set_vertex(j,v);
      vj->set_cell(cnew);
      v->set_cell(c);
      c->neighbor(i)->set_neighbor(c->neighbor(i)->index(c),cnew);
      c->set_neighbor(i,cnew);
      cnew->set_vertex(i,v);
      cnew->set_neighbor(j,c);

      // the code here duplicates a large part of the code 
      // of Triangulation_ds_cell_circulator_3

      VMCell* ctmp = c->neighbor( next_around_edge(i,j) );

      VMCell* cprev = c;
      VMCell* cnewprev = cnew;

      while ( ctmp != c ) {
	// the current cell is duplicated. vertices and neighbors i and j
	// are updated during the traversal.
	// uses the field prev of the circulator
	i = ctmp->index(vi);
	j = ctmp->index(vj);
	cnew = create_cell(ctmp);
	// v will become vertex j of c
	// and vertex i of cnew
	ctmp->set_vertex(j,v);
	ctmp->neighbor(i)->set_neighbor(ctmp->neighbor(i)->index(ctmp),cnew);
	ctmp->set_neighbor(i,cnew);
	cnew->set_vertex(i,v);
	cnew->set_neighbor(j,ctmp);

	// neighbor relations of all cells are used
	// to find relations between new cells
	cnew->set_neighbor(ctmp->index(cprev),cnewprev);
	cnewprev->set_neighbor(cprev->index(ctmp),cnew);

	cnewprev = cnew;
	cprev = ctmp;
	ctmp = ctmp->neighbor( next_around_edge(i,j) );
      }
      cnew = c->neighbor(c->index(vi));
      cnew->set_neighbor(c->index(cprev),cnewprev);
      cnewprev->set_neighbor(cprev->index(c),cnew);
      break;
    }

  case 2:
    {
      assert( is_edge(c,i,j) );
      int k=3-i-j; // index of the third vertex of the facet
      VMCell* d = c->neighbor(k);
      int kd = d->index(c);
      int id = d->index(c->vertex(i));
      int jd = d->index(c->vertex(j));

      cnew = create_cell();
      cnew->set_vertex(i,c->vertex(i)); 
      c->vertex(i)->set_cell(cnew);
      cnew->set_vertex(j,v);
      cnew->set_vertex(k,c->vertex(k));
      c->set_vertex(i,v);

      dnew = create_cell();
      dnew->set_vertex(id,d->vertex(id));
      // d->vertex(id)->cell() is cnew OK
      dnew->set_vertex(jd,v);
      dnew->set_vertex(kd,d->vertex(kd));
      d->set_vertex(id,v);

      cnew->set_neighbor(i,c);
      VMCell* nj = c->neighbor(j);
      cnew->set_neighbor(j,nj);
      nj->set_neighbor(nj->index(c),cnew);
      c->set_neighbor(j,cnew);
      cnew->set_neighbor(k,dnew);

      dnew->set_neighbor(id,d);
      nj = d->neighbor(jd);
      dnew->set_neighbor(jd,nj);
      nj->set_neighbor(nj->index(d),dnew);
      d->set_neighbor(jd,dnew);
      dnew->set_neighbor(kd,cnew);

      v->set_cell(cnew);
      break;
    }

  case 1:
    {
      assert( is_edge(c,i,j) );
      cnew = create_cell(v,c->vertex(1),0,0,
			 c->neighbor(0),c,0,0);
      c->vertex(1)->set_cell(cnew);
      c->set_vertex(1,v);
      c->neighbor(0)->set_neighbor(1,cnew);
      c->set_neighbor(0,cnew);

      v->set_cell(cnew); 
      break;
    }
  }
  set_number_of_vertices(number_of_vertices() +1);

  return v;
}// end insert_in_edge

VMVertex* VolumeMesh::insert_increase_dimension(VMVertex* v, // new vertex
					      VMVertex* star,
					      bool reorient) 
  // star = vertex from which we triangulate the facet of the
  // incremented dimension  
  // ( geometrically : star = infinite vertex )
  // = 0 only used to insert the 1st vertex (dimension -2 to dimension -1)
  // changes the dimension
  // if (reorient) the orientation of the cells is modified
{  // insert()
  if ( v == 0 ) 
    v = create_vertex();

  VMCell* c;
  VMCell* d;
  VMCell* e;
  int i, j;

  switch ( dimension() ) {

  case -2:
    // insertion of the first vertex
    // ( geometrically : infinite vertex )
    {
      assert( number_of_vertices() == 0);
      set_number_of_vertices( 1 );
      set_dimension( -1 );

      c = create_cell( v, 0, 0, 0, 0, 0, 0, 0 );
      v->set_cell(c);
      break;
    }

  case -1:
    // insertion of the second vertex
    // ( geometrically : first finite vertex )
    {
      assert( star != 0 );
      assert( is_vertex(star) ); 
      // this precondition is not expensive when there is only one vertex!

      set_number_of_vertices( number_of_vertices()+1 );
      set_dimension( dimension()+1 );

      d = create_cell( v, 0, 0, 0,
		       star->cell(), 0, 0, 0 );
      v->set_cell(d);
      star->cell()->set_neighbor(0,d);
      break;
    }

  case 0:
    // insertion of the third vertex
    // ( geometrically : second finite vertex )
    {
      assert( star != 0 );
      assert( is_vertex(star) );

      set_number_of_vertices( number_of_vertices()+1 );
      set_dimension( dimension()+1 );

      c = star->cell();
      d = c->neighbor(0);

      if (reorient) {
	c->set_vertex(0,d->vertex(0));
	c->set_vertex(1,star);
	c->set_neighbor(1,d);
	d->set_vertex(1,d->vertex(0));
	d->set_vertex(0,v);
	d->set_neighbor(0,c);
	e = create_cell( star, v, 0, 0,
			 d, c, 0, 0 );
	c->set_neighbor(0,e);
	d->set_neighbor(1,e);
      }
      else {
	c->set_vertex(1,d->vertex(0));
	d->set_vertex(1,v);
	d->set_neighbor(1,c);
	e = create_cell( v, star, 0, 0,
			 c, d, 0, 0 );
	c->set_neighbor(1,e);
	d->set_neighbor(0,e);
      }
	
      v->set_cell(d);
      break;
    }

  case 1:
    // general case : 4th vertex ( geometrically : 3rd finite vertex )
    // degenerate cases geometrically : 1st non collinear vertex
    {
      assert( star != 0 );
      assert( is_vertex(star) );

      set_number_of_vertices( number_of_vertices()+1 );
      set_dimension( dimension()+1 );
      // this is set now, so that it becomes allowed to reorient
      // new facets or cells by iterating on them (otherwise the
      // dimension is to small)

      c = star->cell();
      i = c->index(star); // i== 0 or 1
      j = (1-i);
      d = c->neighbor(j);
	
      c->set_vertex(2,v);

      e = c->neighbor(i);
      VMCell* cnew = c;
      VMCell* enew=0;
	
      while( e != d ){
	enew = create_cell( );
	enew->set_vertex(i,e->vertex(j));
	enew->set_vertex(j,e->vertex(i));
	enew->set_vertex(2,star);
	  
	enew->set_neighbor(i,cnew);
	cnew->set_neighbor(j,enew); 
	// false at the first iteration of the loop where it should
	// be neighbor 2 
	// it is corrected after the loop
	enew->set_neighbor(2,e);
	// neighbor j will be set during next iteration of the loop
	  
	e->set_vertex(2,v);
	e->set_neighbor(2,enew);

	e = e->neighbor(i);
	cnew = enew;
      }
	
      d->set_vertex(2,v);
      d->set_neighbor(2,enew);
      enew->set_neighbor(j,d);
	
      // corrections for star->cell() :
      c = star->cell();
      c->set_neighbor(2,c->neighbor(i)->neighbor(2));
      c->set_neighbor(j,d);
	
      v->set_cell(d);
	
      if (reorient) {
	// reorientation of all the cells
	VMVertex* vtmp;
	VMCell* ctmp;
	VMFacetIterator<VolumeMesh> fit = facets_begin();
	  
	while(fit != facets_end()) {
	  vtmp = (*fit).first->vertex(1);
	  (*fit).first->set_vertex(1,(*fit).first->vertex(0));
	  (*fit).first->set_vertex(0,vtmp);
	    
	  ctmp = (*fit).first->neighbor(1);
	  (*fit).first->set_neighbor(1,(*fit).first->neighbor(0));
	  (*fit).first->set_neighbor(0,ctmp);
	    
	  ++fit;
	}
      }
      break;
    }

  case 2:
    // general case : 5th vertex ( geometrically : 4th finite vertex )
    // degenerate cases : geometrically 1st non coplanar vertex
    {
      assert( star != 0 );
      assert( is_vertex(star) );

      set_number_of_vertices( number_of_vertices()+1 );
      set_dimension( dimension()+1 );

      VMCell* old_cells = list_of_cells().d_next; 
      // used to store the beginning of the list of cells,
      // which will be past end for the list of new cell
      // in order to be able to traverse only the new cells 
      // to find the missing neighbors (we know that new cells are put
      // at the beginning of the list).
	
      VMCell* cnew;
      VMCellIterator<VolumeMesh> it = cells_begin(); 
      // allowed since the dimension has already been set to 3

      v->set_cell(&(*it)); // ok since there is at list one ``cell''
      while (it != cells_end()) {
	it->set_vertex(3,v);
	if ( ! it->has_vertex(star) ) {
	  cnew = create_cell( it->vertex(0),it->vertex(2),
			      it->vertex(1),star,
			      0,0,0,&(*it));
	  it->set_neighbor(3,cnew);
	}
	++it;
      }

      it = cells_begin(); 
      VMCell* n;
      VMCell* c;
      // traversal of the new cells only, to add missing neighbors
      while ( &(*it) != old_cells ) {
	n = it->neighbor(3); // opposite to star
	for ( int i=0; i<3; i++ ) {
	  int j;
	  if ( i==0 ) j=0;
	  else j=3-i; // vertex 1 and vertex 2 are always switched when
	  // creating a new cell (see above)
	  if ( ( c = n->neighbor(i)->neighbor(3) ) != 0 ) {
	    // i.e. star is not a vertex of n->neighbor(i)
	    it->set_neighbor(j,c);
	    // opposite relation will be set when it arrives on c
	    // this avoids to look for the correct index 
	    // and to test whether *it already has neighbor i
	  }
	  else {
	    // star is a vertex of n->neighbor(i)
	    it->set_neighbor(j,n->neighbor(i));
	    n->neighbor(i)->set_neighbor(3,&(*it)); // neighbor opposite to v
	  }
	}
	++it;
      }
	
      // reorientation of all the cells
      if (reorient) {
	VMVertex* vtmp;
	VMCell* ctmp;
	it = cells_begin();
	  
	while ( it != cells_end() ) {
	  vtmp = it->vertex(1);
	  it->set_vertex(1,it->vertex(0));
	  it->set_vertex(0,vtmp);

	  ctmp = it->neighbor(1);
	  it->set_neighbor(1,it->neighbor(0));
	  it->set_neighbor(0,ctmp);
	    
	  ++it;
	}
      }
    }
  }// end switch
    
  return v;
}

VMCell* VolumeMesh::
create_star_3(VMVertex* v, VMCell* c, int li, VMCell * prev_c, VMVertex * prev_v)
{
  //  cout << " creating cell number: " << d_num++ << endl;
    assert( dimension() == 3);
    unsigned char i[3] = {(li+1)&3, (li+2)&3, (li+3)&3};
    if ( (li&1) == 0 )
      std::swap(i[0], i[1]);

    VMVertex *v0 = c->vertex(i[0]);
    VMVertex *v1 = c->vertex(i[1]);
    VMVertex *v2 = c->vertex(i[2]);
    VMCell * cnew = create_cell(v0, v1, v2, v);
    v0->set_cell(cnew);
    v1->set_cell(cnew);
    v2->set_cell(cnew);
    VMCell * c_li = c->neighbor(li);
    cnew->set_neighbor(3, c_li);
    c_li->set_neighbor(c_li->index(c), cnew);

    // Look for the other three neighbors of cnew.
    for (int ii=0; ii<3; ii++) {
      if ( prev_v == c->vertex(i[ii]) ) {
        cnew->set_neighbor(ii, prev_c);
        continue;
      }
      // Indices of the vertices of cnew such that i[ii],j1,j2,li positive.
      int j1 = next_around_edge(i[ii],li);
      int j2 = 6-li-i[ii]-j1;
      const VMVertex *vj1 = c->vertex(j1);
      const VMVertex *vj2 = c->vertex(j2);
      VMCell *cur = c;
      VMCell *n = c->neighbor(i[ii]);
      // turn around the oriented edge j1 j2
      while ( n->get_conflict_flag() > 0) {
	// The main loop is free from orientation problems.
	// It remains only before and after...  It could probably be done.
	assert( n != c );
        if (n->neighbor(0) != cur &&
            n->vertex(0) != vj1 && n->vertex(0) != vj2)
          cur = n, n = n->neighbor(0);
        else
        if (n->neighbor(1) != cur &&
            n->vertex(1) != vj1 && n->vertex(1) != vj2)
          cur = n, n = n->neighbor(1);
        else
        if (n->neighbor(2) != cur && 
            n->vertex(2) != vj1 && n->vertex(2) != vj2)
          cur = n, n = n->neighbor(2);
        else
	  cur = n, n = n->neighbor(3);
      }
      // Now n is outside region, cur is inside.
      n->set_conflict_flag(0);
      VMCell *nnn;
      int kkk;
      if (n->has_neighbor(cur, kkk)) {
	// Neighbor relation is reciprocal, ie
	// the cell we are looking for is not yet created.
        VMVertex * next_prev;
        if (kkk != 0 && n->vertex(0) != vj1 && n->vertex(0) != vj2)
           next_prev = n->vertex(0);
        else if (kkk != 1 && n->vertex(1) != vj1 && n->vertex(1) != vj2)
           next_prev = n->vertex(1);
        else if (kkk != 2 && n->vertex(2) != vj1 && n->vertex(2) != vj2)
           next_prev = n->vertex(2);
        else
           next_prev = n->vertex(3);

	nnn = create_star_3(v, cur, cur->index(n), cnew, next_prev);
      }
      else
      {
        // else the cell we are looking for was already created
        int jj1 = n->index( vj1 );
        int jj2 = n->index( vj2 );
        nnn = n->neighbor( next_around_edge(jj2,jj1) );
      }
      cnew->set_neighbor(ii, nnn);
    }

    return cnew;
}

VMCell* VolumeMesh::create_star_2(VMVertex* v, VMCell* c, int li )
{
  assert( dimension() == 2 );
  VMCell* cnew;

  // i1 i2 such that v,i1,i2 positive
  int i1=ccw(li);
  // traversal of the boundary of region in ccw order to create all
  // the new facets
  VMCell* bound = c;
  VMVertex* v1 = c->vertex(i1);
  int ind = c->neighbor(li)->index(c); // to be able to find the
                                       // first cell that will be created 
  VMCell* cur;
  VMCell* pnew = 0;
  do {
    cur = bound;
    // turn around v2 until we reach the boundary of region
    while ( cur->neighbor(cw(i1))->get_conflict_flag() > 0 ) {
      // neighbor in conflict
      cur = cur->neighbor(cw(i1));
      i1 = cur->index( v1 );
    }
    cur->neighbor(cw(i1))->set_conflict_flag(0);
    // here cur has an edge on the boundary of region
    cnew = create_cell( v, v1, cur->vertex( ccw(i1) ), 0,
			cur->neighbor(cw(i1)), 0, pnew, 0);
    cur->neighbor(cw(i1))->set_neighbor
      ( cur->neighbor(cw(i1))->index(cur), cnew );
    // pnew is 0 at the first iteration
    v1->set_cell(cnew);
    //pnew->set_neighbor( cw(pnew->index(v1)), cnew );
    if (pnew) { pnew->set_neighbor( 1, cnew );}

    bound = cur;
    i1 = ccw(i1);
    v1 = bound->vertex(i1);
    pnew = cnew;
    //} while ( ( bound != c ) || ( li != cw(i1) ) );
  } while ( v1 != c->vertex(ccw(li)) );
  // missing neighbors between the first and the last created cells
  cur = c->neighbor(li)->neighbor(ind); // first created cell
  cnew->set_neighbor( 1, cur );
  cur->set_neighbor( 2, cnew );
  return cnew;
}

void VolumeMesh::incident_cells(VMVertex* v, std::set<VMCell*> & cells, VMCell* c) const
{
  assert( v != 0 );
  assert( is_vertex(v) );

  if ( dimension() < 3 )
      return;

  if ( c == 0 )
    c = v->cell();
  else
    assert( c->has_vertex(v) );

  if ( cells.find( c ) != cells.end() )
    return; // c was already found

  cells.insert( c );
      
  for ( int j=0; j<4; j++ )
    if ( j != c->index(v) )
      incident_cells( v, cells, c->neighbor(j) );
}
  
void VolumeMesh::
incident_vertices(VMVertex* v, std::set<VMVertex*> & vertices, VMCell* c) const
{
  assert( v != 0 );
  assert( is_vertex(v) );
      
  if ( number_of_vertices() < 2 )
      return;

  if ( c == 0 )
    c = v->cell();
  else
    assert( c->has_vertex(v) );

  int d = dimension();
  int j;
  int found = 0;
  for ( j=0; j <= d; j++ ) {
    if ( j != c->index(v) ) {
      if ( vertices.find( c->vertex(j) ) == vertices.end() )
	vertices.insert( c->vertex(j) );
      else
	found++; // c->vertex(j) was already found 
    }
  }
  if ( found == 3 )
      return; // c was already visited
      
  for ( j=0; j <= d; j++ )
    if ( j != c->index(v) )
      incident_vertices( v, vertices, c->neighbor(j) );
}

bool VolumeMesh::is_valid(bool verbose, int level ) const
{
  switch ( dimension() ) {
  case 3:
    {
      int vertex_count;
      if ( ! count_vertices(vertex_count,verbose,level) )
	  return false;
      if ( number_of_vertices() != vertex_count ) {
	if (verbose)
	    std::cerr << "false number of vertices" << std::endl;
	assert(false);
	return false;
      }

      int cell_count;
      if ( ! count_cells(cell_count,verbose,level) )
	  return false;
      int edge_count;
      if ( ! count_edges(edge_count,verbose,level) )
	  return false;
      int facet_count;
      if ( ! count_facets(facet_count,verbose,level) )
	  return false;

      // Euler relation 
      if ( cell_count - facet_count + edge_count - vertex_count != 0 ) {
	if (verbose)
	    std::cerr << "Euler relation unsatisfied" << std::endl;
	assert(false);
	return false;
      }

      break;
    }
  case 2:
    {
      int vertex_count;
      if ( ! count_vertices(vertex_count,verbose,level) )
	  return false;
      if ( number_of_vertices() != vertex_count ) {
	if (verbose)
	    std::cerr << "false number of vertices" << std::endl;
	assert(false);
	return false;
      }

      int edge_count;
      if ( ! count_edges(edge_count,verbose,level) )
	  return false;
      // Euler for edges
      if ( edge_count != 3 * vertex_count - 6 ) {
	if (verbose)
	    std::cerr << "Euler relation unsatisfied - edges/vertices"
		      << std::endl;
	assert(false);
	return false;
      }

      int facet_count;
      if ( ! count_facets(facet_count,verbose,level) )
	  return false;
      // Euler for facets
      if ( facet_count != 2 * vertex_count - 4 ) {
	if (verbose)
	    std::cerr << "Euler relation unsatisfied - facets/vertices"
		      << std::endl;
	assert(false);
	return false;
      }
      break;
    }
  case 1:
    {
      int vertex_count;
      if ( ! count_vertices(vertex_count,verbose,level) )
	  return false;
      if ( number_of_vertices() != vertex_count ) {
	if (verbose)
	    std::cerr << "false number of vertices" << std::endl;
	assert(false);
	return false;
      }
      int edge_count;
      if ( ! count_edges(edge_count,verbose,level) )
	  return false;
      // Euler for edges
      if ( edge_count != vertex_count ) {
	if (verbose)
	    std::cerr << "false number of edges" << std::endl;
	assert(false);
	return false;
      }
      break;
    }
  case 0:
    {
      if ( number_of_vertices() < 2 ) {
	if (verbose)
	    std::cerr << "less than 2 vertices but dimension 0" << std::endl;
	assert(false);
	return false;
      }
      // no break; continue
    }
  case -1:
    {
      if ( number_of_vertices() < 1 ) {
	if (verbose)
	  std::cerr << "no vertex but dimension -1" << std::endl;
	assert(false);
	return false;
      }
      // vertex count
      int vertex_count;
      if ( ! count_vertices(vertex_count,verbose,level) )
	return false;
      if ( number_of_vertices() != vertex_count ) {
	if (verbose)
	  std::cerr << "false number of vertices" << std::endl;
	assert(false);
	return false;
      }
    } 
  } // end switch
  if (verbose)
      std::cerr << "valid data structure" << std::endl;
  return true;
}

VMVertex* VolumeMesh::copy_mesh(const VolumeMesh & mesh, VMVertex* vert )
  // returns the new vertex corresponding to vert in the new tds 
{
  assert( vert == 0 || mesh.is_vertex(vert) );

  clear();

  int n = mesh.number_of_vertices();
  set_number_of_vertices(n);
  set_dimension(mesh.dimension());

  if (n == 0) return vert;

  // Create the vertices.
  // the vertices must be indexed by their order of creation so
  // that when reread from file, the orders of vertices are the
  // same - important for remove 
  std::vector<VMVertex*> TV(n);
  int i = 0;

  for (VMVertexIterator<VolumeMesh> vit = mesh.vertices_begin();
       vit != mesh.vertices_end(); ++vit)
    TV[i++] = &*vit; 
  
  assert( i == n ); 
  std::sort(TV.begin(), TV.end(), Vertex_mesh_compare_order_of_creation<VMVertex*>()); 

  std::map< VMVertex*, VMVertex* > V;
  std::map< VMCell*, VMCell* > F;

  for (i=0; i <= n-1; i++) {
    V[ TV[i] ] = create_vertex();
    *V[ TV[i] ] = *TV[i];
  }

  // Create the cells.
  for (VMCell* cit = mesh.d_list_of_cells.d_next;
       cit != mesh.past_end_cell(); cit = cit->d_next) {
      F[&(*cit)] = create_cell(&*cit);
      F[&(*cit)]->set_vertices(V[cit->vertex(0)],
			       V[cit->vertex(1)],
			       V[cit->vertex(2)],
			       V[cit->vertex(3)]);
  }

  // Link the vertices to a cell.
  for (VMVertexIterator<VolumeMesh> vit2 = mesh.vertices_begin();
       vit2 != mesh.vertices_end(); ++vit2)
    V[&(*vit2)]->set_cell( F[vit2->cell()] );

  // Hook neighbor pointers of the cells.
  for (VMCell* cit2 = mesh.d_list_of_cells.d_next;
       cit2 != mesh.past_end_cell(); cit2 = cit2->d_next) {
    for (int j = 0; j < 4; j++)
      F[&(*cit2)]->set_neighbor(j, F[cit2->neighbor(j)] );
  }

  assert( is_valid() );

  return (vert != 0) ? V[vert] : 0;
}

void VolumeMesh::clear()
{
  std::vector<VMVertex*> Vertices;
  clear_cells_only(Vertices);

  // deletion of the vertices
  for ( std::vector<VMVertex*>::iterator vit = Vertices.begin();
       vit != Vertices.end(); ++vit )
    delete *vit;

  set_number_of_vertices(0);
  set_dimension(-2);
}


void VolumeMesh::clear_cells_only(std::vector<VMVertex *> & Vertices)
{
  assert(d_list_of_temporary_free_cells.d_next == &d_list_of_temporary_free_cells);
  assert(d_list_of_temporary_free_cells.d_previous == &d_list_of_temporary_free_cells);

  VMCell *it;

  // Delete the cells in the free_list.
  for (it = d_list_of_free_cells.d_next; it != &d_list_of_free_cells;
       it = d_list_of_free_cells.d_next) {
      remove_cell_from_list(it);
      delete it;
  }

  if (number_of_vertices() == 0) {
    // the list of cells must be cleared even in this case
    for (it = d_list_of_cells.d_next; it != past_end_cell();
         it = d_list_of_cells.d_next) {
      remove_cell_from_list(it);
      delete it;
    }

    // then _list_of_cells points on itself, nothing more to do
    set_dimension(-2);
    return;
  }

  // We must save all vertices because we're going to delete the cells.
  Vertices.reserve(number_of_vertices());

  // deletion of the cells
  // does not use the cell iterator to work in any dimension
  for (it = d_list_of_cells.d_next; it != past_end_cell();
       it = d_list_of_cells.d_next)
  {
    // We save the vertices to delete them after.
    // We use the same trick as the Vertex_iterator.
    for (int i=0; i<=std::max(0,dimension()); i++)
      if (it->vertex(i)->cell() == it)
        Vertices.push_back(&(*it->vertex(i)));
    remove_cell_from_list(it);
    delete it;
  }

  // then _list_of_cells points on itself, nothing more to do
  assert(d_list_of_cells.d_next == &d_list_of_cells);
  assert(d_list_of_cells.d_previous ==&d_list_of_cells);
}

bool VolumeMesh::count_vertices(int & i, bool verbose, int level) const
  // counts AND checks the validity
{
  i = 0;

  for (VMVertexIterator<VolumeMesh> it = vertices_begin(); it != vertices_end(); ++it) {
    if ( ! it->is_valid(verbose,level) ) {
      if (verbose)
	  std::cerr << "invalid vertex" << std::endl;
      assert(false);
      return false;
    }
    ++i;
  }
  return true;
} 

bool VolumeMesh::count_facets(int & i, bool verbose, int level) const
  // counts but does not check
{
  i = 0;

  for (VMFacetIterator<VolumeMesh> it = facets_begin(); it != facets_end(); ++it) {
    if ( ! (*it).first->is_valid(dimension(),verbose, level) ) {
      if (verbose)
	  std::cerr << "invalid facet" << std::endl;
      assert(false);
      return false;
    }
    ++i;
  }
  return true;
}

bool VolumeMesh::count_edges(int & i, bool verbose, int level) const
  // counts but does not check
{
  i = 0;

  for (VMEdgeIterator<VolumeMesh> it = edges_begin(); it != edges_end(); ++it) {
    if ( ! (*it).first->is_valid(dimension(),verbose, level) ) {
      if (verbose) std::cerr << "invalid edge" << std::endl;
      assert(false);
      return false;
    }
    ++i;
  }
  return true;
}

bool VolumeMesh::count_cells(int & i, bool verbose, int level) const
  // counts AND checks the validity
{
  i = 0;

  for (VMCellIterator<VolumeMesh> it = cells_begin(); it != cells_end(); ++it) {
    if ( ! it->is_valid(dimension(),verbose, level) ) {
      if (verbose)
	  std::cerr << "invalid cell" << std::endl;
      assert(false);
      return false;
    }
    ++i;
  }
  return true;
}

}

#endif
