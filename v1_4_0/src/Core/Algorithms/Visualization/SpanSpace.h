/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  SpanSpace.h: The Span Data type
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SpanSpace_h
#define SpanSpace_h 

#include <Core/Persistent/Persistent.h>
#include <Core/Datatypes/Datatype.h>
//#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/Field.h>
//#include <Core/Containers/LockingHandle.h>


namespace SCIRun {
    
  // SpanPoint

  template <class T, class Index>
  struct SpanPoint {
    T min;
    T max;
    Index index;
	
    SpanPoint(){}
    SpanPoint(T min, T max, Index i) : min(min), max(max), index(i) {}
  };

  // SpanSpace
  
  class SpanSpaceBase {
  public:
    SpanSpaceBase() {}
    virtual ~SpanSpaceBase() {}
  };
  
  template <class T, class Index>
  class SpanSpace : public SpanSpaceBase {
  public:
    typedef SpanPoint<T,Index>  span_point;
    Array1<span_point> span;
    
  public:
    SpanSpace() {}
    ~SpanSpace() {}

    template<class Field> void init( Field *);
    void swap( span_point &, span_point &);
    void select_min( span_point p[], int n );
    void select_max( span_point p[], int n );
    
  };
  
  template <class T,class Index>
  void SpanSpace<T,Index>::swap (span_point &a, span_point &b)
  {
    span_point t = a;
    a = b;
    b = t;
  }
  
  // FUNCTIONS
  
  template <class T,class Index>
  void SpanSpace<T,Index>::select_min( span_point p[], int n )
  {
    if ( n < 2 )
      return;
    
    int k = n/2;
    int l = 0;
    int r = n-1;
    while ( r > l ) {
      int mid = (l+r)/2;
      if ( p[l].min > p[mid].min ) swap( p[l], p[mid] );
      if ( p[l].min > p[r].min ) swap( p[l], p[r] );
      if ( p[mid].min > p[r].min ) swap( p[mid], p[r] );
      
      T v = p[r].min;
      
      int i,j;
      for( i=l-1, j=r; ; ) {
	while ( v > p[++i].min );
	while ( p[--j].min > v );
	if ( i >= j ) 
	  break;
	
	swap( p[i], p[j] );
      }
      swap( p[i], p[r]) ;
      
      if ( i >= k ) r = i-1;
      if ( i <= k ) l = i+1;
    }
    
    select_max( p, n/2 );
    select_max( p+n/2+1, (n-1)/2 );
  }
  
  
  template <class T, class Index>
  void SpanSpace<T,Index>::select_max( span_point p[], int n )
  {
    if ( n < 2 )
      return;
    
    int k = n/2;
    int l = 0;
    int r = n-1;
    while ( r > l ) {
      int mid = (l+r)/2;
      if ( p[l].max > p[mid].max ) swap( p[l], p[mid] );
      if ( p[l].max > p[r].max ) swap( p[l], p[r] );
      if ( p[mid].max > p[r].max ) swap( p[mid], p[r] );
      
      T v = p[r].max;
      
      int i,j;
      for( i=l-1, j=r; ; ) {
	while ( v > p[++i].max );
	while ( p[--j].max > v );
	if ( i >= j ) 
	  break;
	
	swap( p[i], p[j] );
      }
      swap( p[i], p[r]) ;
      
      if ( i >= k ) r = i-1;
      if ( i <= k ) l = i+1;
    }
    
    select_min( p, n/2 );
    select_min( p+n/2+1, (n-1)/2 );
  }
  
  
  
  template<class T, class Index>
    template <class Field> 
    void SpanSpace<T,Index>::init(Field *field)
    {
      typedef typename Field::mesh_type  mesh_type;
      //typedef typename mesh_type::Cell::index_type cell_index_type;
      
      typename Field::mesh_handle_type mesh = field->get_typed_mesh();

      typename mesh_type::Cell::iterator cell; mesh->begin(cell);
      typename mesh_type::Cell::iterator cell_end; mesh->end(cell_end);
      typename mesh_type::Node::array_type nodes;

      for ( ; cell != cell_end; ++cell) {
	mesh->get_nodes( nodes, *cell );

	// compute  min max of cell
	T min, max;
	min = max = field->value(nodes[0]);
	for (unsigned i=1; i<nodes.size(); i++) {
	  T v = field->value(nodes[i]);
	  if ( v < min ) min = v;
	  else if ( v > max ) max = v;
	}
	
	if ( min < max ) // ignore cells with min == max
	  span.add( SpanPoint<T,Index>(min, max, *cell));
      }
      
      // init kd-tree
      select_min( &span[0], span.size() );    
    }
  
  
  template <class T,class Index>
    void Pio( Piostream &, SpanSpace<T,Index> &) {}
  
  template <class T,class Index> 
  const string find_type_name(SpanSpace<T,Index>*)
  {
    static const string name = string("SpanSpace") + FTNS 
      + find_type_name((T*)0) + FTNM 
      + /*find_type_name((Index *)0)*/ "CellIndex<?>" + FTNE;
    return name;
  }

/*   class SpanUniverse : public Datatype  */
/*   { */
/*   public: */
/*     Array1< SpanSpaceBase *> space; */
/*     FieldHandle field; */
/*     int generation; */
/*     BBox bbox; */
/*     int dx, dy; */
    
/*   public: */
/*     SpanUniverse( FieldHandle field) : field(field) {} */
/*     virtual ~SpanUniverse() {} */
    
/*     void add( SpanSpaceBase *base) { space.add(base); } */
    
/*     // Persistent representation */
/*     virtual void io(Piostream&) {}; */
/*     static PersistentTypeID type_id; */
/*   }; */
  
/*   typedef LockingHandle<SpanUniverse> SpanUniverseHandle;  */
  
} // namespace SCIRun


#endif // SpanSpace_h
