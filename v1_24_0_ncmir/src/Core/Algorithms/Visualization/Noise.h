/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


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
#include <Core/Util/DynamicLoader.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>

namespace SCIRun {

// NoiseBase

class NoiseAlg : public DynamicAlgoBase {
public:
  NoiseAlg();
  virtual ~NoiseAlg();
  
  virtual void release() = 0;
  virtual void set_field( Field * ) = 0;
  virtual GeomHandle search( double ival, bool field, bool geom) = 0;

  //! support the dynamically compiled algorithm concept
  static const string& get_h_file_path();
  static CompileInfoHandle get_compile_info(const TypeDescription *td,
					    bool cell_centered_p,
					    bool face_centered_p);

  FieldHandle trisurf_;
  MatrixHandle interpolant_;

  FieldHandle get_field();
  MatrixHandle get_interpolant();
};

// Noise<T>

template <class Tesselator>
class Noise : public NoiseAlg
{
protected:
  typedef typename Tesselator::field_type       field_type;
  typedef typename field_type::value_type       value_type;
  typedef typename field_type::mesh_type::Elem::index_type cell_index_type;
  typedef SpanPoint<value_type, cell_index_type>      span_point;

  typename SpanSpace<value_type,cell_index_type>::handle_type space_;
  Tesselator *tess_;
  double v;
  
public:
  Noise() : space_(0), tess_(0) {}
  virtual ~Noise() { release(); }
  
  virtual void release();
  virtual void set_field( Field *);
  GeomHandle search( double ival, bool field, bool geom );
  
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


template <class Tesselator>
class NoiseFace : public Noise<Tesselator>
{
  typedef typename Tesselator::field_type       field_type;
  typedef typename field_type::value_type       value_type;
  typedef typename field_type::mesh_type::Elem::index_type cell_index_type;
  typedef SpanPoint<value_type, cell_index_type>      span_point;

public:
  virtual ~NoiseFace() {}
  
  virtual void set_field( Field *);
};


template <class Tesselator>
class NoiseCell : public Noise<Tesselator>
{
  typedef typename Tesselator::field_type       field_type;
  typedef typename field_type::value_type       value_type;
  typedef typename field_type::mesh_type::Elem::index_type cell_index_type;
  typedef SpanPoint<value_type, cell_index_type>      span_point;

public:
  virtual ~NoiseCell() {}
  
  virtual void set_field( Field *);
};


// Noise

template<class Tesselator>
void Noise<Tesselator>::release()
{
  if ( tess_ ) { delete tess_; tess_ = 0; }
}


template<class Tesselator>
void Noise<Tesselator>::set_field( Field *f )
{
  if ( field_type *field = dynamic_cast<field_type *>(f) ) {
    if ( tess_ ) delete tess_;
    tess_ = new Tesselator( field );
    if ( !field->get_property( "spanspace", space_ ) )
    {
      space_ = scinew SpanSpace<value_type, cell_index_type>;
      space_->init( field );
      field->set_property( "spanspace", space_, true );
    }
  }
}


template<class Tesselator>
void NoiseFace<Tesselator>::set_field( Field *f )
{
  if ( field_type *field = dynamic_cast<field_type *>(f) )
  {
    if ( this->tess_ ) delete this->tess_;
    this->tess_ = new Tesselator( field );
    if ( !field->get_property( "spanspace", this->space_ ) )
    {
      SpanSpaceFace<value_type, cell_index_type> *space =
	scinew SpanSpaceFace<value_type, cell_index_type>;
      space->init_face( field );
      this->space_ = space;
      field->set_property( "spanspace", this->space_, true );
    }
  }
}

template<class Tesselator>
void NoiseCell<Tesselator>::set_field( Field *f )
{
  if ( field_type *field = dynamic_cast<field_type *>(f) )
  {
    if ( this->tess_ ) delete this->tess_;
    this->tess_ = new Tesselator( field );
    if ( !field->get_property( "spanspace", this->space_ ) )
    {
      SpanSpaceCell<value_type, cell_index_type> *space =
	scinew SpanSpaceCell<value_type, cell_index_type>;
      space->init_cell( field );
      this->space_ = space;
      field->set_property( "spanspace", this->space_, true );
    }
  }
}


template <class Tesselator> 
int Noise<Tesselator>::count( )
{
  int counter = 0;
  count_min_max( &space_->span[0], space_->span.size(), counter );
  
  return counter;
}


template <class Tesselator>
GeomHandle Noise<Tesselator>::search( double iso, bool bf, bool bg )
{
  v = iso;
  
  int n = count();
  tess_->reset(n, bf, bg);
  
  search_min_max( &space_->span[0], space_->span.size() );

  trisurf_ = tess_->get_field(iso);
  interpolant_ = tess_->get_interpolant();
  return tess_->get_geom();
}

template <class Tesselator> 
void Noise<Tesselator>::search_min_max( span_point p[], int n )
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


template <class Tesselator> 
void Noise<Tesselator>::search_max_min( span_point p[], int n )
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


template <class Tesselator> 
void Noise<Tesselator>::search_min( span_point p[], int n)
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


template <class Tesselator> 
void Noise<Tesselator>::search_max( span_point p[], int n )
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

template <class Tesselator> 
void Noise<Tesselator>::collect( span_point p[], int n )
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


template <class Tesselator> 
void Noise<Tesselator>::count_min_max( span_point p[], int n, int &counter)
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


template <class Tesselator> 
void Noise<Tesselator>::count_max_min( span_point p[], int n, int &counter)
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


template <class Tesselator> 
void Noise<Tesselator>::count_min( span_point p[], int n, int &counter )
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


template <class Tesselator> 
void Noise<Tesselator>::count_max( span_point p[], int n, int &counter )
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



template <class Tesselator>
void Noise<Tesselator>::count_collect( int n, int &counter )
{
  if ( n <= 0 )
    return;
  
  counter += n;
}


} // namespace SCIRun

#endif // Noise_h
