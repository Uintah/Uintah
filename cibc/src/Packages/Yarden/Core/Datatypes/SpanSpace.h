
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

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array3.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Containers/LockingHandle.h>


namespace Yarden {
    
using namespace SCIRun;
    
    // HEADERS

    // SpanPoint

    template <class T>
      struct SpanPoint {
	T min;
	T max;
	int index;
	
	SpanPoint(){}
	SpanPoint(T min, T max, int i) : min(min), max(max), index(i) {}
      };

    // SpanSpace

    class SpanSpaceBase {
    public:
      SpanSpaceBase() {}
      virtual ~SpanSpaceBase() {}
    };

    template <class T>
      class SpanSpace : public SpanSpaceBase
      {
      public:
	Array1< SpanPoint<T> > span;
	
      public:
	SpanSpace() {}
	~SpanSpace() {}
	
	void swap( SpanPoint<T> &, SpanPoint<T> &);
	void select_min( SpanPoint<T> p[], int n );
	void select_max( SpanPoint<T> p[], int n );
	
      };
    
    // SpanSpaceBuild
    //     Build a SpaceSpace. Templated on a ScalarField

    template <class T, class F>
      class SpanSpaceBuild : public SpanSpace<T>
      {
      public:
	SpanSpaceBuild( F *field);
	~SpanSpaceBuild() {}
      };

    // SpanSpaceBuildUG
    //     Build a SpaceSpace from Unstructured ScalarField

    class SpanSpaceBuildUG : public SpanSpace<double>
      {
      public:
	SpanSpaceBuildUG( ScalarFieldUG *field);
	~SpanSpaceBuildUG() {}
      };


    template <class T>
      void SpanSpace<T>::swap (SpanPoint<T> &a, SpanPoint<T> &b)
      {
	SpanPoint<T> t = a;
	a = b;
	b = t;
      }

    // FUNCTIONS

    template <class T>
      void SpanSpace<T>::select_min( SpanPoint<T> p[], int n )
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
      
    
    template <class T>
      void SpanSpace<T>::select_max( SpanPoint<T> p[], int n )
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
    
    
    
    template <class T,class F> 
      SpanSpaceBuild<T,F>::SpanSpaceBuild ( F *field)
      {
	Array3<T> &grid = field->grid;
	int pos = 0;
	
	for ( int k=0; k<grid.dim3()-1; k++, pos += grid.dim1() )
	  for ( int j=0; j<grid.dim2()-1; j++, pos++ )
	    for ( int i=0; i<grid.dim1()-1; i++, pos++  ) {
	      
	      T min, max, v;
	      min = max = grid(i,j,k);
	      
	      v = grid(i+1,j,k);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	      
	      v = grid(i,j+1,k);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	      
	      v = grid(i+1,j+1,k);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	      
	      v = grid(i,j,k+1);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	      
	      v = grid(i+1,j,k+1);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
		  
	      v = grid(i,j+1,k+1);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	      
	      v = grid(i+1,j+1,k+1);
	      if ( v < min ) min = v;
	      else if ( v > max ) max = v;
	      
	      if ( min < max ) {
		span.add(SpanPoint<T>( min, max, pos ));
	      }
	    }
	
	select_min( &span[0], span.size() );
      }



    class SpanUniverse : public Datatype 
      {
      public:
	Array1< SpanSpaceBase *> space;
	ScalarFieldHandle field;
	int generation;
	BBox bbox;
	int dx, dy;
	
      public:
	SpanUniverse( ScalarFieldHandle field) : field(field) {}
	virtual ~SpanUniverse() {}

	void add( SpanSpaceBase *base) { space.add(base); }

	// Persistent representation
	virtual void io(Piostream&) {};
	static PersistentTypeID type_id;
      };
    
    typedef LockingHandle<SpanUniverse> SpanUniverseHandle; 
  
} // End namespace Yarden


#endif 
