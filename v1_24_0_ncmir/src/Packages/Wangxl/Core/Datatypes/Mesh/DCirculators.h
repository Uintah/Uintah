#ifndef SCI_Wangxl_Datatypes_Mesh_DCirculators_h
#define SCI_Wangxl_Datatypes_Mesh_DCirculators_h

#include <Packages/Wangxl/Core/Datatypes/Mesh/VMCirculators.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DVertex.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DEdge.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DFacet.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/DCell.h>
#include <Packages/Wangxl/Core/Datatypes/Mesh/CirculatorBase.h>
namespace Wangxl {

using namespace SCIRun;

class Delaunay;

template<class Mesh>
class DCellCirculator
  : public Bidirectional_circulator_base<DCell, ptrdiff_t, size_t>
{
public:
  /*  typedef typename Tds::Cell Ctds;
  typedef typename Tds::Cell_circulator Circulator_base;

  typedef Triangulation_3<Gt,Tds> Triangulation;

  typedef typename Triangulation::Cell Cell;
  typedef typename Triangulation::Vertex Vertex;
  typedef typename Triangulation::Edge Edge;
  typedef typename Triangulation::Vertex_handle Vertex_handle;
  typedef typename Triangulation::Cell_handle Cell_handle;

  typedef DCellCirculator<Gt,Tds> Cell_circulator;
  */
  DCellCirculator() : d_ci(), d_dln(0) {}

  DCellCirculator(const Delaunay* dln, DCell* c, int s, int t)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), c, s, t ), d_dln(const_cast<Delaunay *>(dln)) {}

  DCellCirculator(const Delaunay* dln, const DEdge& e)
    : d_ci( &(const_cast<Delaunay *>(dln)->d_mesh), e.first,
	    e.second, e.third ), d_dln(const_cast<Delaunay *>(dln)) {}
  
  DCellCirculator(const Delaunay* dln, DCell* c, int s, int t,
		  DCell* start)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), c, s, t, start ),
      d_dln(const_cast<Delaunay*>(dln)) {}

  DCellCirculator(const Delaunay* dln, const DEdge & e, DCell* start)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), 
	    e.first, e.second, e.third, 
	    start), d_dln(const_cast<Delaunay *>(dln)) {}
  
  DCellCirculator(const DCellCirculator & ccir) : d_ci(ccir.d_ci), d_dln(ccir.d_dln) {}

  DCellCirculator& operator=(const DCellCirculator & ccir)
  {
    d_ci = ccir.d_ci;
    d_dln = ccir.d_dln;
    return *this;
  }
  
  bool operator==(const DCellCirculator & ccir) const
  {
    return ( d_ci == ccir.d_ci);
  }
  
  bool operator!=(const DCellCirculator & ccir)
  {
    return ( !(*this == ccir) );
  }

  DCellCirculator& operator++()
  {
    ++d_ci;
    return *this;
  }

  DCellCirculator& operator--()
  {
    --d_ci;
    return *this;
  }

  DCellCirculator operator++(int)
  {
    DCellCirculator tmp(*this);
    ++(*this);
    return tmp;
  }
        
  DCellCirculator operator--(int)
  {
    DCellCirculator tmp(*this);
    --(*this);
    return tmp;
  }

  DCell& operator*() const
  {
    return (DCell &)(*d_ci);
  }

  DCell* operator->() const
  {
    return (DCell*)( &(*d_ci) );
  }

private: 
  VMCellCirculator<Mesh> d_ci;
  Delaunay* d_dln;
};

template<class Mesh>
class DFacetCirculator
  : public Bidirectional_circulator_base<DFacet, ptrdiff_t, size_t>
{
public:
  /*  typedef typename Tds::Cell VMCell;
  typedef typename Tds::Facet_circulator Circulator_base;

  typedef Triangulation_3<Gt,Tds> Triangulation;

  typedef typename Triangulation::Cell Cell;
  typedef typename Triangulation::Vertex Vertex;
  typedef typename Triangulation::Edge Edge;
  typedef typename Triangulation::Facet Facet;
  typedef typename Triangulation::Vertex_handle Vertex_handle;
  typedef typename Triangulation::DCellHandle DCellHandle;

  typedef DFacetCirculator<Gt,Tds> Facet_circulator;
  */
  DFacetCirculator() : d_ci(), d_dln(NULL) {}
  
  DFacetCirculator(const Delaunay* dln, DCell* c, int s, int t)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), c, s, t ),
d_dln(const_cast<Delaunay*>(dln)) {}

  DFacetCirculator(const Delaunay* dln, const DEdge& e)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), e.first,
	    e.second, e.third ), d_dln(const_cast<Delaunay *>(dln)){}

  DFacetCirculator(const Delaunay* dln, DCell* c, int s, int t, const DFacet& start)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), c, s, t,
	    std::make_pair(start.first, start.second) ),
d_dln(const_cast<Delaunay*>(dln)) {}

  DFacetCirculator(const Delaunay* dln, const DEdge& e, const DFacet& start)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh), e.first, e.second, e.third, 
	    std::make_pair(start.first, start.second) ),
    d_dln(const_cast<Delaunay*>(dln)) {}
 
  DFacetCirculator(const Delaunay* dln, DCell* c, int s, int t, DCell* start, int f)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh),
	    c, s, t, start, f ), d_dln(const_cast<Delaunay*>(dln)) {}

   DFacetCirculator(const Delaunay* dln, const DEdge& e, DCell* start, int f)
    : d_ci( &(const_cast<Delaunay*>(dln)->d_mesh),
	    e.first, e.second, e.third, start, f ),
     d_dln(const_cast<Delaunay*>(dln)){}
 
   DFacetCirculator(const DFacetCirculator& ccir) : d_ci(ccir.d_ci), d_dln(ccir.d_dln) {}

  DFacetCirculator& operator=(const DFacetCirculator& ccir)
  {
    d_ci = ccir.d_ci;
    d_dln = ccir.d_dln;
    return *this;
  }
  
  bool operator==(const DFacetCirculator& ccir) const
  {
    return ( d_ci == ccir.d_ci);
  }

  bool operator!=(const DFacetCirculator& ccir)
  {
    return ( !(*this == ccir) );
  }

  DFacetCirculator& operator++()
    {
      ++d_ci;
      return *this;
    }

  DFacetCirculator& operator--()
    {
      --d_ci;
      return *this;
    }

  DFacetCirculator operator++(int)
  {
    DFacetCirculator tmp(*this);
    ++(*this);
    return tmp;
  }
        
  DFacetCirculator operator--(int)
    {
      DFacetCirculator tmp(*this);
      --(*this);
      return tmp;
    }

  DFacet operator*() const
  {
    return std::make_pair( (*d_ci).first, (*d_ci).second ) ;
  }

private:
  VMFacetCirculator<Mesh> d_ci;
  Delaunay* d_dln;
};

}

#endif  // CGALD_DLNIANGULATION_CIRCULATORS_3_H
