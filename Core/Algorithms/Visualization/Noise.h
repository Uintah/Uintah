/*
 *  Noise.h: A Near Optimal IsoSurface Extraction
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

#include <Core/Algorithms/Visualization/SpanSpace.h>

namespace SCIRun {

  // NoiseBase
  
  class NoiseAlg {
  public:
    NoiseAlg() {}
    virtual ~NoiseAlg() {}
    
    virtual void set_field( Field * ) = 0;
    virtual GeomObj* search( double ) = 0;
  };

  template < class AI >
  class NoiseBase : public NoiseAlg {
  protected:
    AI *ai_;
  public:
    NoiseBase() {}
    NoiseBase( AI *ai) : ai_(ai) {}
    virtual ~NoiseBase() {}
    
  };
  
  // Noise<T, AI>
  
  template <class AI, class Tesselator>
  class Noise : public NoiseBase<AI> {
    typedef typename Tesselator::field_type       field_type;
    typedef typename field_type::value_type       value_type;
    typedef typename field_type::mesh_type::cell_index cell_index;
    typedef SpanPoint<value_type,cell_index>      span_point;
  private:
    SpanSpace<value_type,cell_index> *space_;
    Tesselator *tess_;
    double v;
    
  public:
    Noise() {}
    Noise( AI *ai) : NoiseBase<AI>(ai), space_(0), tess_(0) {}
    virtual ~Noise() {}
    
    virtual void set_field( Field *);
    GeomObj *search( double );
    
    int count();
    
    void search_min_max( span_point p[], int n );
    void search_max_min( span_point p[], int n );
    void search_min( span_point p[], int n );
    void search_max( span_point p[], int n );
    void collect( span_point p[], int n );
    
    void count_min_max( span_point p[], int n, int & );
    void count_max_min( span_point p[], int n, int & );
    void count_min( span_point p[], int n, int & );
    void count_max( span_point p[], int n, int & );
    void count_collect( int n, int & );
    
  };
  
  /*
   * NoiseBase class
   */
  
  
  // Noise
  
  template<class AI, class Tesselator>
  void Noise<AI, Tesselator>::set_field( Field *f )
  {
    if ( field_type *field = dynamic_cast<field_type *>(f) ) {
      if ( tess_ ) delete tess_;
      tess_ = new Tesselator( field );
      if ( !field->get( "spanspace", space_ ) ) {
	space_ = scinew SpanSpace<value_type,cell_index>;
	space_->init( field );
	field->store( "spanspace", space_ );
      }
    }
  }

  template <class AI, class Tesselator> 
  int Noise<AI, Tesselator>::count( )
  {
    int counter = 0;
    count_min_max( &space_->span[0], space_->span.size(), counter );
    
    return counter;
  }
    
    
  template <class AI, class Tesselator>
  GeomObj *Noise<AI, Tesselator>::search( double iso )
  {
    v = iso;
    
    int n = count();
    tess_->reset(n);
    
    search_min_max( &space_->span[0], space_->span.size() );
    
    return tess_->get_geom();
  }
  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::search_min_max( span_point p[], int n )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p+n/2;
    
    if ( v > point->min ) {
      if ( point->max > v )
	tess_->extract(point->index, v );
      
      search_max_min( point+1, (n-1)/2);
      search_max( p, n/2 );
    }
    else
      search_max_min( p, n/2 );
  }
  
  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::search_max_min( span_point p[], int n )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p + n/2;
    
    if ( point->max > v ) {
      if ( v > point->min )
	tess_->extract(point->index, v );
      
      search_min_max( p, n/2 );
      search_min( point+1, (n-1)/2 );
    }
    else
      search_min_max( point+1, (n-1)/2 );
    
  }
  
  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::search_min( span_point p[], int n)
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p + n/2;
    
    if ( v > point->min  ) {
      tess_->extract(point->index, v );
      
      // Right Son.
      if ( n > 2) {	
	span_point *child = point+1+(n-1)/4;
	
	if ( v > child->min  )
	  tess_->extract(child->index, v );
	
	search_min( child+1, ((n-1)/2 -1)/2);
	search_min( point+1, (n-1)/4 );
      }
      
      // Left son: collect all.
      collect( p, n/2 );
    }
    else {
      span_point *child = p+n/4;
      if ( v > child->min  )
	tess_->extract(child->index, v );
      
      search_min( p+n/4+1, (n/2-1)/2 );
      search_min( p, n/4 );
    }
    
  }

    
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::search_max( span_point p[], int n )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p + n/2;
    
    if ( point->max > v ) {
      tess_->extract(point->index, v );
      
      if ( n > 1 ) {
	span_point *child = p+n/4;
	
	if ( child->max > v )
	  tess_->extract(child->index, v );
	
	search_max( p, n/4 );
	search_max( child+1, (n-2)/4 );
      }
      collect( point+1, (n-1)/2 );
    }
    else
      if ( n > 1 ) {
	span_point *child = point+1+(n-1)/4;
	
	if ( child->max > v )
	  tess_->extract(child->index, v );
	
	search_max( point+1, (n-1)/4 );
	search_max( child+1, (n-3)/4 );
      }
    
  }
    
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::collect( span_point p[], int n )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p;
    for ( int i=0; i<n; i++, point++ )
      tess_->extract(point->index, v );
  }
  
  /*
   * Count
   */
  
  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::count_min_max( span_point p[], int n, int &counter )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p+n/2;
    
    if ( point->min <= v ) {
      if ( point->max > v )
	counter++;
      
      count_max_min( point+1, (n-1)/2, counter );
      count_max( p, n/2, counter );
    }
    else
      count_max_min( p, n/2, counter );
  }
  
  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::count_max_min( span_point p[], int n, int &counter )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p + n/2;
    
    if ( point->max >= v ) {
      if ( v > point->min )
	counter++;
      
      count_min_max( p, n/2, counter );
      count_min( point+1, (n-1)/2, counter );
    }
    else
      count_min_max( point+1, (n-1)/2, counter );
  }
  
  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::count_min( span_point p[], int n, int &counter )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p + n/2;
    
    if ( point->min <= v ) {
      counter++;
      
      // Right Son.
      if ( n > 2) {	
	span_point *child = point+1+(n-1)/4;
	
	if ( v > child->min  )
	  counter++;
	
	count_min( child+1, ((n-1)/2 -1)/2, counter);
	count_min( point+1, (n-1)/4, counter );
      }
      
      // Left son: collect all.
      count_collect( n/2, counter );
    }
    else {
      span_point *child = p+n/4;
      if ( v > child->min  )
	counter++;
      
      count_min( p+n/4+1, (n/2-1)/2, counter );
      count_min( p, n/4, counter );
    }
  }

  
  template <class AI, class Tesselator> 
  void Noise<AI, Tesselator>::count_max( span_point p[], int n, int &counter )
  {
    if ( n <= 0 )
      return;
    
    span_point *point = p + n/2;
    
    if ( point->max >= v ) {
      counter++;
      
      if ( n > 1 ) {
	span_point *child = p+n/4;
	
	if ( child->max > v )
	  counter++;
	
	count_max( p, n/4, counter );
	count_max( child+1, (n-2)/4, counter );
      }
      count_collect( (n-1)/2, counter );
    }
    else
      if ( n > 1 ) {
	span_point *child = point+1+(n-1)/4;
	
	if ( child->max > v )
	  counter++;
	
	count_max( point+1, (n-1)/4, counter );
	count_max( child+1, (n-3)/4, counter );
      }
  }
  
  
  
  template <class AI, class Tesselator>
  void Noise<AI, Tesselator>::count_collect( int n, int &counter )
  {
    if ( n <= 0 )
      return;
    
    counter += n;
  }
  

} // namespace SCIRun

#endif Noise_h
