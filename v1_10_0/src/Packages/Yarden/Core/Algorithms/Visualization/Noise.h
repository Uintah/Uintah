/*
 *  Noise.h
 *      View Depended Iso Surface Extraction
 *      for Structures Grids (Bricks)
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2000
 *
 *  Copyright (C) 2000 SCI Group
 */


#ifndef Noise_h
#define Noise_h

namespace Yarden {

using namespace SCIRun;
    
    // NoiseBase

    template < class AI >
    class NoiseBase 
      {
      protected:
	AI *ai;
      public:
	NoiseBase( AI *ai) : ai(ai) {}
	virtual ~NoiseBase() {}
	
	virtual GeomGroup *extract( double) = 0;
      };
    
    // Noise<T, AI>
    
    template <class T, class I, class AI>
      class Noise : public NoiseBase<AI>
      {
      private:
	SpanSpace< T > *space;
	I *interp;
	double v;

      public:
	Noise(  SpanSpace<T> *, I *interp, AI *);
	~Noise();
	
	int count();
	GeomGroup *extract( double );

	void search_min_max( SpanPoint<T> p[], int n );
	void search_max_min( SpanPoint<T> p[], int n );
	void search_min( SpanPoint<T> p[], int n );
	void search_max( SpanPoint<T> p[], int n );
	void collect( SpanPoint<T> p[], int n );

	void count_min_max( SpanPoint<T> p[], int n, int & );
	void count_max_min( SpanPoint<T> p[], int n, int & );
	void count_min( SpanPoint<T> p[], int n, int & );
	void count_max( SpanPoint<T> p[], int n, int & );
	void count_collect( int n, int & );
	
      };

    /*
     * NoiseBase class
     */


    // Noise
    
    template <class T, class I, class AI>
      Noise<T,I,AI>::Noise( SpanSpace<T> *space, I *interp, AI *ai ) 
      :   NoiseBase<AI>(ai), space(space), interp(interp)
      {
      }
    
    template <class T, class I, class AI>
      Noise<T,I,AI>::~Noise()
      {
      }

    template <class T, class I,  class AI> 
      int Noise<T,I,AI>::count( )
      {
	int counter = 0;
	count_min_max( &space->span[0], space->span.size(), counter );
	
	return counter;
      }
    
    
    template <class T, class I,  class AI>
      GeomGroup *Noise<T,I,AI>::extract( double iso )
      {
	v = iso;

	int n = count();
	interp->reset(n);

	search_min_max( &space->span[0], space->span.size() );
	
	return interp->getGeom();
      }

    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::search_min_max( SpanPoint<T> p[], int n )
      {
	if ( n <= 0 )
	  return;
	
	SpanPoint<T> *point = p+n/2;

	if ( v > point->min ) {
	  if ( point->max > v )
	    interp->extract(point->index, v );

	  search_max_min( point+1, (n-1)/2);
	  search_max( p, n/2 );
	}
	else
	  search_max_min( p, n/2 );
      }


    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::search_max_min( SpanPoint<T> p[], int n )
      {
	if ( n <= 0 )
	  return;

	SpanPoint<T> *point = p + n/2;
	
	if ( point->max > v ) {
	  if ( v > point->min )
	    interp->extract(point->index, v );
	  
	  search_min_max( p, n/2 );
	  search_min( point+1, (n-1)/2 );
	}
	else
	  search_min_max( point+1, (n-1)/2 );

      }
    
    
    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::search_min( SpanPoint<T> p[], int n)
      {
	if ( n <= 0 )
	  return;

	SpanPoint<T> *point = p + n/2;

	if ( v > point->min  ) {
	  interp->extract(point->index, v );

	  // Right Son.
	  if ( n > 2) {	
	    SpanPoint<T> *child = point+1+(n-1)/4;

	    if ( v > child->min  )
	      interp->extract(child->index, v );

	    search_min( child+1, ((n-1)/2 -1)/2);
	    search_min( point+1, (n-1)/4 );
	  }

	  // Left son: collect all.
	  collect( p, n/2 );
	}
	else {
	  SpanPoint<T> *child = p+n/4;
	  if ( v > child->min  )
	    interp->extract(child->index, v );

	  search_min( p+n/4+1, (n/2-1)/2 );
	  search_min( p, n/4 );
	}

      }

    
    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::search_max( SpanPoint<T> p[], int n )
      {
	if ( n <= 0 )
	  return;

	SpanPoint<T> *point = p + n/2;
    
	if ( point->max > v ) {
	  interp->extract(point->index, v );

	  if ( n > 1 ) {
	    SpanPoint<T> *child = p+n/4;

	    if ( child->max > v )
	      interp->extract(child->index, v );

	    search_max( p, n/4 );
	    search_max( child+1, (n-2)/4 );
	  }
	  collect( point+1, (n-1)/2 );
	}
	else
	  if ( n > 1 ) {
	    SpanPoint<T> *child = point+1+(n-1)/4;

	    if ( child->max > v )
	      interp->extract(child->index, v );

	    search_max( point+1, (n-1)/4 );
	    search_max( child+1, (n-3)/4 );
	  }

      }
    

    
    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::collect( SpanPoint<T> p[], int n )
      {
	if ( n <= 0 )
	  return;

	SpanPoint<T> *point = p;
	for ( int i=0; i<n; i++, point++ )
	  interp->extract(point->index, v );
      }

    /*
     * Count
     */
    
    
    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::count_min_max( SpanPoint<T> p[], int n, int &counter )
      {
	if ( n <= 0 )
	  return;
	
	SpanPoint<T> *point = p+n/2;

	if ( point->min <= v ) {
	  if ( point->max > v )
	    counter++;
	  
	  count_max_min( point+1, (n-1)/2, counter );
	  count_max( p, n/2, counter );
	}
	else
	  count_max_min( p, n/2, counter );
      }
    

    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::count_max_min( SpanPoint<T> p[], int n, int &counter )
      {
	if ( n <= 0 )
	  return;
	
	SpanPoint<T> *point = p + n/2;
    
	if ( point->max >= v ) {
	  if ( v > point->min )
	    counter++;
	
	  count_min_max( p, n/2, counter );
	  count_min( point+1, (n-1)/2, counter );
	}
	else
	  count_min_max( point+1, (n-1)/2, counter );
      }
    
    
    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::count_min( SpanPoint<T> p[], int n, int &counter )
      {
	if ( n <= 0 )
	  return;

	SpanPoint<T> *point = p + n/2;

	if ( point->min <= v ) {
	  counter++;

	  // Right Son.
	  if ( n > 2) {	
	    SpanPoint<T> *child = point+1+(n-1)/4;

	    if ( v > child->min  )
	      counter++;

	    count_min( child+1, ((n-1)/2 -1)/2, counter);
	    count_min( point+1, (n-1)/4, counter );
	  }

	  // Left son: collect all.
	  count_collect( n/2, counter );
	}
	else {
	  SpanPoint<T> *child = p+n/4;
	  if ( v > child->min  )
	    counter++;

	  count_min( p+n/4+1, (n/2-1)/2, counter );
	  count_min( p, n/4, counter );
	}
      }

    
    template <class T, class I,  class AI> 
      void Noise<T,I,AI>::count_max( SpanPoint<T> p[], int n, int &counter )
      {
	if ( n <= 0 )
	  return;

	SpanPoint<T> *point = p + n/2;
    
	if ( point->max >= v ) {
	  counter++;

	  if ( n > 1 ) {
	    SpanPoint<T> *child = p+n/4;

	    if ( child->max > v )
	      counter++;

	    count_max( p, n/4, counter );
	    count_max( child+1, (n-2)/4, counter );
	  }
	  count_collect( (n-1)/2, counter );
	}
	else
	  if ( n > 1 ) {
	    SpanPoint<T> *child = point+1+(n-1)/4;

	    if ( child->max > v )
	      counter++;

	    count_max( point+1, (n-1)/4, counter );
	    count_max( child+1, (n-3)/4, counter );
	  }
      }
    

    
    template <class T, class I, class AI> 
      void Noise<T,I,AI>::count_collect( int n, int &counter )
      {
	if ( n <= 0 )
	  return;
	
	counter += n;
      }


} // End namespace Yarden

#endif
