#ifndef SCI_Wangxl_Datatypes_Mesh_VMIterators_h
#define SCI_Wangxl_Datatypes_Mesh_VMIterators_h

//#include <utility>

#include <Packages/Wangxl/Core/Datatypes/Mesh/Triple.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMFacet.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCirculators.h>

namespace Wangxl {

using namespace SCIRun;

class VMVertex;
class VMCell;

template<class Mesh>
class VMCellIterator
{
public:
  //  typedef Tds_                                  Tds;
  typedef VMCell                    value_type;
  typedef VMCell*                  pointer;
  typedef VMCell&                  reference;
  typedef size_t                           size_type;
  typedef ptrdiff_t                        difference_type;
  typedef std::bidirectional_iterator_tag       iterator_category;

  //  typedef typename Tds::Cell                    Cell;
  //  typedef Triangulation_ds_cell_iterator_3<Tds> Cell_iterator;

  // CONSTRUCTORS

  VMCellIterator() {}
  
  VMCellIterator(const Mesh * mesh)
    : d_mesh(const_cast<Mesh *>(mesh))
    {
      if ( d_mesh->dimension() < 3 )
	  pos = d_mesh->past_end_cell(); // there is no cell yet
      else
	  pos = d_mesh->list_of_cells().d_next; // first cell of the list
    }

  // used to initialize the past-the end iterator
  VMCellIterator(const Mesh* mesh, int)
    : d_mesh(const_cast<Mesh *>(mesh))
    {
      pos = d_mesh->past_end_cell();
    }
  
  // OPERATORS
  
  VMCellIterator & operator=(const VMCellIterator & ci)
    {
      pos = ci.pos;
      d_mesh = ci.d_mesh;
      return *this;
    }

  bool operator==(const VMCellIterator & ci) const
    {
      return pos == ci.pos && d_mesh == ci.d_mesh;
    }
  
  bool operator!=(const VMCellIterator& ci)
    {
      return !(*this == ci);
    }
  
  VMCellIterator & operator++()
    {
      pos = pos->d_next;
      return *this;
    }

  VMCellIterator & operator--()
    {
      pos = pos->d_previous;
      return *this;
    }
        
  VMCellIterator operator++(int)
    {
      VMCellIterator tmp(*this);
      ++(*this);
      return tmp;
    }
  
  VMCellIterator operator--(int)
    {
      VMCellIterator tmp(*this);
      --(*this);
      return tmp;
    }
        
  VMCell & operator*() const
    {
      return *pos;
    }
    
  VMCell* operator->() const
    {
      return pos;
    }

private:
  Mesh*  d_mesh;
  VMCell* pos;
};

template<class Mesh>
class VMVertexIterator
{
// traverses the list of cells and reports for each cell 
// the vertices whose cell() is the current cell

public:
  typedef VMVertex                   value_type;
  typedef VMVertex*                 pointer;
  typedef VMVertex&                 reference;
  typedef size_t                            size_type;
  typedef ptrdiff_t                         difference_type;
  typedef std::bidirectional_iterator_tag        iterator_category;

  /*  typedef typename Tds::Vertex                    Vertex;
  typedef typename Tds::Cell                      Cell;
  typedef Triangulation_ds_vertex_iterator_3<Tds> Vertex_iterator;
  */
  VMVertexIterator() : d_mesh(0), pos(0), index(0) {}
  
  VMVertexIterator(const Mesh* mesh)
    : d_mesh(const_cast<Mesh *>(mesh)), index(0)
    {
      if ( d_mesh->number_of_vertices() == 0 )
	pos = d_mesh->past_end_cell(); 
      else { 
	pos = d_mesh->list_of_cells().d_next; 
	while ( (pos != d_mesh->past_end_cell())
             && (pos != pos->vertex(index)->cell()) )
	  increment();
      }
    }
  
  // used to initialize the past-the end iterator
  VMVertexIterator(const Mesh* mesh, int)
    : d_mesh(const_cast<Mesh *>(mesh)), index(0)
    {
      pos = d_mesh->past_end_cell();
    }
  
  VMVertexIterator& operator++()
  {
    do {
      increment();
    } while ( (pos != d_mesh->past_end_cell())
           && (pos != pos->vertex(index)->cell()) );
    return *this;
  }
    
  VMVertexIterator& operator--()
  {
    do {
      if (index == 0) {
	// all the vertices of the current cell have been examined
	int d = d_mesh->dimension();
	if ( d >= 0 )
	    index = d;
	pos = pos->d_previous;
      }
      else
	  index--;
    } while ( pos != d_mesh->past_end_cell()
	   && pos != pos->vertex(index)->cell() );
    return *this;
  }
    
  VMVertexIterator operator++(int)
    {
      VMVertexIterator tmp(*this);
      ++(*this);
      return tmp;
    }
    
  VMVertexIterator operator--(int)
    {
      VMVertexIterator tmp(*this);
      --(*this);
      return tmp;
    }
    
  bool operator==(const VMVertexIterator& vi) const
    {
      return d_mesh == vi.d_mesh && pos == vi.pos && index == vi.index;
    }
    
  bool operator!=(const VMVertexIterator& vi) const
    {
      return !(*this == vi);
    }
    
  VMVertex & operator*() const
    {
      return *(pos->vertex(index));
    }
    
  VMVertex* operator->() const
    {
      return pos->vertex(index);
    }

private:
  Mesh*  d_mesh;
  VMCell* pos; // current "cell". Even if the dimension is <3 when 
             // there is no true cell yet.
  int index; // index of the current vertex in the current cell

  void 
  increment()
  {
    if (index >= d_mesh->dimension()) {
      // all the vertices of the current cell have been examined
      index = 0;
      pos = pos->d_next;
    }
    // be careful : index should always be 0 when pos = past_end_cell
    else
	index++;
  }
};

template<class Mesh>
class VMFacetIterator
{
// traverses the list of cells and report for each cell 
// the vertices whose cell() is the current cell

public:
  typedef VMFacet                    value_type;
  typedef VMFacet*                  pointer;
  typedef VMFacet&                  reference;
  typedef size_t                            size_type;
  typedef ptrdiff_t                         difference_type;
  typedef std::bidirectional_iterator_tag        iterator_category;

  VMFacetIterator() : d_mesh(0), pos(0), index(0) {}
  
  VMFacetIterator(const Mesh * mesh)
    : d_mesh(const_cast<Mesh *>(mesh)), index(0)
    {
      switch ( d_mesh->dimension() ) {
      case 2:
	pos = d_mesh->list_of_cells().d_next; 
	index = 3;
	return;
      case 3:
	pos = d_mesh->list_of_cells().d_next; 
	while ( // useless (pos != _tds->past_end_cell()) &&
	       // there must be at least one facet
	       pos->neighbor(index) < pos ) {
	  increment();
	}
	return;
      default:
	pos = d_mesh->past_end_cell();
	return;
      }
    }
  
  // used to initialize the past-the end iterator
  VMFacetIterator(const Mesh* mesh, int)
    : d_mesh(const_cast<Mesh *>(mesh)), index(0)
    {
	pos = d_mesh->past_end_cell();
	if (d_mesh->dimension() == 2)
	    index = 3;
    }
  
  VMFacetIterator& operator++()
  {
    if (d_mesh->dimension() < 2)
	return *this;

    if (d_mesh->dimension() == 3) {
      do {
	increment();
      } while ( pos != d_mesh->past_end_cell()
	     && pos > pos->neighbor(index) );
      // reports a facet when the current cell has a pointer inferior
      // to the pointer of the neighbor cell
      return *this;
    }

    pos = pos->d_next; // dimension 2
    return *this;
  }
    
  VMFacetIterator& operator--()
  {
    if (d_mesh->dimension() < 2)
	return *this;
    
    if ( d_mesh->dimension() == 2 ) {
      pos = pos->d_previous; // index remains 3
      return *this;
    }

    // dimension 3
    do{
      if (index == 0) {
	// all the facets of the current cell have been examined
	index = 3;
	pos = pos->d_previous;
      }
      else
	  index--;
    } while ( pos != d_mesh->past_end_cell()
	   && pos > pos->neighbor(index) );
    // reports a facet when the current cell has a pointer inferior
    // to the pointer of the neighbor cell
    return *this;
  }
    
  VMFacetIterator operator++(int)
    {
      VMFacetIterator tmp(*this);
      ++(*this);
      return tmp;
    }
    
  VMFacetIterator operator--(int)
    {
      VMFacetIterator tmp(*this);
      --(*this);
      return tmp;
    }
    
  bool operator==(const VMFacetIterator& fi) const
    {
      return d_mesh == fi.d_mesh && pos == fi.pos && index == fi.index;
    }
    
  bool operator!=(const VMFacetIterator& fi) const
    {
      return !(*this == fi);
    }
    
  VMFacet operator*() const
    {
      // case pos == NULL should not be accessed, there is no facet
      // when dimension <2 
      return std::make_pair(pos, index);
    }
    
private:
  Mesh*  d_mesh;
  VMCell* pos; // current "cell". Even if the dimension is <3 when 
              // there is no true cell yet.
  int index; // index of the current facet in the current cell

  void increment()
  {
    if (index == 3) {
      // all the feces of the current cell have been examined
      index = 0;
      pos = pos->d_next;
    }
    // be careful : index should always be 0 when pos = past_end_cell
    else
	index++;
  }

};

template<class Mesh>
class VMEdgeIterator
{
// traverses the list of cells and report for each cell 
// the vertices whose cell() is the current cell

public:
  typedef VMEdge                     value_type;
  typedef VMEdge*                    pointer;
  typedef VMEdge&                    reference;
  typedef size_t                     size_type;
  typedef ptrdiff_t                  difference_type;
  typedef std::bidirectional_iterator_tag         iterator_category;

  //  typedef typename Tds::Cell                      Cell;
  //  typedef typename Tds::Edge                      Edge;
  //  typedef Triangulation_ds_edge_iterator_3<Tds>   Edge_iterator;
  //  typedef Triangulation_ds_cell_circulator_3<Tds> Cell_circulator;

  VMEdgeIterator() : d_mesh(0), pos(0), b(0), e(1) {}
  
  VMEdgeIterator(const Mesh * mesh)
    : d_mesh(const_cast<Mesh *>(mesh)), b(0), e(1)
    {
      switch ( d_mesh->dimension() ) {
      case 1:
	{
	  pos = d_mesh->list_of_cells().d_next;
	  return;
	}
      case 2:
	{
	  pos = d_mesh->list_of_cells().d_next; 
	  while ( // useless (pos != _tds->past_end_cell()) && 
		 // there must be at least one edge
		 ( pos->neighbor(3-b-e) < pos) ) {
	    increment2();
	  }
	  return;
	}
      case 3:
	{
	  pos = d_mesh->list_of_cells().d_next;
	  bool notfound = true;
	  while ( // useless (pos != _tds->past_end_cell()) &&
		 // there must be at least one edge
		 notfound ) {
	    VMCellCirculator<Mesh> ccir = d_mesh->incident_cells(pos,b,e);
	    do {
	      ++ccir;
	    } while ( &(*ccir) > pos ); 
	    // loop terminates since it stops at least when ccir = pos
	    if ( &(*ccir) == pos ) // pos is the cell with minimum pointer
	      notfound = false;
	    else
	      increment3();
	  }
	  return;
	}
      default:
	{
	  pos = d_mesh->past_end_cell() ; 
	  return;
	}
      }
    }
  
  // used to initialize the past-the end iterator
  VMEdgeIterator(const Mesh* mesh, int)
    : d_mesh(const_cast<Mesh *>(mesh)), b(0), e(1)
    {
	pos = mesh->past_end_cell();
    }
  
  VMEdgeIterator& operator++()
  {
    switch ( d_mesh->dimension() ) {
    case 1:
      {
	pos = pos->d_next;
	break;
      }
    case 2:
      {
	do {
	  increment2();
	} while ( pos != d_mesh->past_end_cell() && 
		  pos > pos->neighbor(3-b-e) );
	break;
      }
    case 3:
      {
	bool notfound = true;
	do {
	  increment3();
	  if (pos != d_mesh->past_end_cell()) {
	    VMCellCirculator<Mesh> ccir = d_mesh->incident_cells(pos,b,e);
	    do {
	      ++ccir;
	    } while ( &(*ccir) > pos );
	    if ( &(*ccir) == pos ) // pos is the cell with minimum pointer
	      notfound = false;
	  }
	  else {
	    b=0; e=1;
	  }
	} while ( pos != d_mesh->past_end_cell() && notfound );
	break;
      }
    default:
      {
	return *this;
      }
    }
    return *this;
  }
    
  VMEdgeIterator& operator--()
  {
    switch ( d_mesh->dimension() ) {
    case 1:
      {
	pos = pos->d_previous; // b, e remain 0, 1
	break;
      }
    case 2:
      {
	do {
	  if (b == 0) {
	    b = 2; e = 0;
	    pos = pos->d_previous;
	  }
	  else {
	    b--; 
	    e = b+1; // case b==2, e==0 forbids to write e--
	  }
	} while ( pos != d_mesh->past_end_cell() && 
		  pos > pos->neighbor(3-b-e) );
	break;
      }
    case 3:
      {
	bool notfound = true;
	do {
	  if (b == 0) {
	    if (e == 1) {
	      // all the edges of the current cell have been examined
	      b = 2; e = 3;
	      pos = pos->d_previous;
	    }
	    else
	      e--;
	  }
	  else {
	    if (e == b+1) {
	      b--;
	      e = 3;
	    }
	    else
	      e--;
	  }
	  if (pos != d_mesh->past_end_cell()) {
	    VMCellCirculator<Mesh> ccir = d_mesh->incident_cells(pos,b,e);
	    do {
	      ++ccir;
	    } while ( &(*ccir) > pos );
	    if ( &(*ccir) == pos ) // pos is the cell with minimum pointer
	      notfound = false;
	  }
	  else {
	    b=0; e=1;
	  }
	} while ( pos != d_mesh->past_end_cell() && notfound );
	break;
      }
    default :
      return *this;
    }
    // reports an edge when the current cell has a pointer inferior
    // to the pointer of the neighbor cell
    return *this;
  }
    
  VMEdgeIterator operator++(int)
    {
      VMEdgeIterator tmp(*this);
      ++(*this);
      return tmp;
    }
    
  VMEdgeIterator operator--(int)
    {
      VMEdgeIterator tmp(*this);
      --(*this);
      return tmp;
    }
    
  bool operator==(const VMEdgeIterator& ei) const
    {
      return d_mesh == ei.d_mesh && pos == ei.pos && b == ei.b && e == ei.e;
    }
    
  bool operator!=(const VMEdgeIterator& ei) const
    {
      return !(*this == ei);
    }
    
  VMEdge operator*() const
    {
      return make_triple(pos, b, e);
    }
    
private:
  Mesh*  d_mesh;
  VMCell* pos; // current "cell". Even if the dimension is <3 when 
              // there is no true cell yet.
  int b; // index of the first endpoint of the current edge in the current cell
  int e; // index of the second endpoint of the current edge in the
  // current cell 

  void increment2()
  {
    if (b == 2) { // e == 0
      // all the edges of the current cell have been examined
      b = 0; e = 1;
      pos = pos->d_next;
    }
    // be careful : index should always be 0 when pos = past_end_cell
    else { 
      b++; 
      if ( b == 2 )
	e = 0;
      else // b==1
	e = 2;
    }
  }

  void increment3()
  {
    if (b == 2) { // then e == 3
      // all the edges of the current cell have been examined
      b = 0; e = 1;
      pos = pos->d_next;
    }
    else {
      if (e == 3) {
	b++;
	e = b+1;
      }
      else
	e++;
    }
  }

};

}

#endif

