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


#ifndef Datatypes_LevelField_h
#define Datatypes_LevelField_h


#include "LevelMesh.h"
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>
#include <Core/Util/Assert.h>
#include <Packages/Uintah/Core/Grid/ShareAssignArray3.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Runnable.h>

#include <string>
#include <vector>
using std::string;
using std::vector;

/* //Specialization needed for InterpFunctor<LevelField<Matrix3> > */
/* #include <Core/Datatypes/FieldAlgo.h> */
/* namespace SCIRun { */
/*   using Uintah::Matrix3; */
/*   using Uintah::LevelField; */
/*   template<> InterpFunctor<LevelField<Matrix3> >::InterpFunctor() : */
/*     result_(double(0)) {} */
/* } // namespace SCIRun */

namespace Uintah {

using SCIRun::GenericField;
using SCIRun::LockingHandle;
using SCIRun::Interpolate;

using SCIRun::Thread;
using SCIRun::Mutex;
using SCIRun::Semaphore;
using SCIRun::Runnable;

template <class Data> class LevelField;

template <class Data>
class LevelFieldSFI : public ScalarFieldInterface {
public:
  LevelFieldSFI(const LevelField<Data>* fld) : fld_(fld) {}

  virtual bool compute_min_max(double &minout, double &maxout) const;
  virtual bool interpolate(double &result, const Point &p) const;
  virtual bool interpolate_many(vector<double> &results,
				const vector<Point> &points) const;
private:
  const LevelField<Data>* fld_;
};


// class Data must inherit from Packages/Uintah/Core/Grid/Array3 or
// this will not compile

template <class Data>
class LevelData : public vector<ShareAssignArray3<Data> > 
{
public:
  typedef Data value_type;
  //  typedef  iterator;
  typedef vector<ShareAssignArray3<Data> > parent;


  LevelData():
    vector<ShareAssignArray3<Data> >(), begin_(0), begin_initialized(false),
    end_(0), end_initialized(false) {}
  LevelData(const LevelData& data) :
    vector<ShareAssignArray3<Data> >(data), begin_(0), begin_initialized(false),
    end_(0), end_initialized(false) {}
  virtual ~LevelData(){ }
  
  const value_type &operator[](typename LevelMesh::Cell::index_type idx) const 
{ return parent::operator[](idx.patch_->getLevelIndex())
    [IntVector(idx.i_,idx.j_,idx.k_)]; } 
  const value_type &operator[](typename LevelMesh::Face::index_type idx) const
{ return parent::operator[](0)
    [IntVector(idx.i_,0,0)];}
const value_type &operator[](typename LevelMesh::Edge::index_type idx) const 
{ return parent::operator[](0)
    [IntVector(idx.i_, 0, 0)]; }
const value_type &operator[](typename LevelMesh::Node::index_type idx) const
{ return parent::operator[](idx.patch_->getLevelIndex())
    [IntVector(idx.i_,idx.j_,idx.k_)]; }

value_type &operator[](typename LevelMesh::Cell::index_type idx)
{ return parent::operator[](idx.patch_->getLevelIndex())
    [IntVector(idx.i_,idx.j_,idx.k_)]; } 
value_type &operator[](typename LevelMesh::Face::index_type idx)
{ return parent::operator[](0)
    [IntVector(idx.i_, 0, 0)]; }
value_type &operator[](typename LevelMesh::Edge::index_type idx)
{ return parent::operator[](0)
    [IntVector(idx.i_, 0, 0)]; }
value_type &operator[](typename LevelMesh::Node::index_type idx)
{ return parent::operator[](idx.patch_->getLevelIndex())
    [IntVector(idx.i_,idx.j_,idx.k_)]; }

// These use a pointer to patch and the index to the node or cell to
// get the data out.
// 
// These functions assume idx is the index to an Array3Window that has the
// low index already accounted for.  In other words the index needs no
// shifting.
value_type &get_data_by_patch_and_index(Patch * patch, IntVector idx)
{ return parent::operator[](patch->getLevelIndex())[idx]; }

static const string type_name(int n = -1);
virtual const string get_type_name(int n = -1) const { return type_name(n); }

void resize(const LevelMesh::Node::size_type &) {}
void resize(LevelMesh::Edge::size_type) {}
void resize(LevelMesh::Face::size_type) {}
void resize(const LevelMesh::Cell::size_type &) {}
void resize(int i){ vector<ShareAssignArray3<Data> >::resize(i); }

  class iterator
  {
  public:
    iterator(const vector<ShareAssignArray3<Data> >* data, IntVector index) 
      : it_( (*data)[0].begin() ), vit_(data->begin()), vitend_(data->end())
      {
	for(; vit_ != vitend_; vit_++){
	  IntVector low = (*vit_).getLowIndex();
	  IntVector high = (*vit_).getHighIndex();
	  if( index.x() >= low.x() && index.y() >= low.y() &&
	      index.z() >= low.z() && index.x() < high.x() &&
	      index.y() < high.y() && index.z() < high.z())
	  {
	    it_ = Array3<Data>::const_iterator( &(*vit_), index);
	    return;
	  }
	}
      }
    iterator(const vector<ShareAssignArray3<Data> >* data) 
      : it_( (*data)[0].begin() ), vit_(data->begin()), vitend_(data->end()){}
    iterator(const iterator& iter) 
      : it_(iter.it_), vit_(iter.vit_), vitend_(iter.vitend_){}
    iterator& operator=(const iterator& it)
    { it_ = it.it_; vit_ = it.vit_; vitend_ == it.vitend_; }
    inline bool operator==(const iterator& it)
    { return it_ == it.it_ && vit_ == it.vit_ && vitend_ == it.vitend_;}
    inline bool operator!=(const iterator& it)
    { return !(operator==(it)); }
    inline const Data& operator*() const {return *it_;}
    inline iterator& operator++(){
      if( ++it_ ==  (*vit_).end() ){
	if(++vit_ != vitend_){
	  it_ = (*vit_).begin();
	}
      }
      return *this;
    }
    inline iterator operator++(int){
      iterator result( *this );
      ++(*this);
      return result;
    }
  private:
    typename Array3<Data>::const_iterator it_;
    typename vector<ShareAssignArray3<Data> >::const_iterator vit_;
    typename vector<ShareAssignArray3<Data> >::const_iterator vitend_;
  };
  
inline iterator begin() { 
  if( begin_initialized ) return *begin_;
  else {
    const Array3<Data>& a = parent::operator[](0);
    begin_ =  new iterator(this, a.getLowIndex());
    begin_initialized = true;
    return *begin_;
  }
}
inline iterator end()  {
  if( end_initialized ) return *end_;
  else {
    const Array3<Data>& a = parent::operator[](size()-1);
    iterator it(this, a.getHighIndex() - IntVector(1,1,1));
    end_ = new iterator(++it);
    end_initialized = true;
    return *end_;
  }
}
private:
iterator *begin_;
bool begin_initialized;
iterator *end_;
bool end_initialized;
};

    
template <class Data>
const string
LevelData<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) +
      FTNS + "Array3" + FTNS + type_name(1) + FTNE + " " + FTNE;
    return name;
  }
  else if (n == 0)
  {
    return "LevelData";
  }
  else
  {
    return find_type_name((Data *)0);
  }
}

template <class Data>
class LevelField : public GenericField< LevelMesh, LevelData<Data>  > 
{ 

public:

  LevelField() :
    GenericField<LevelMesh, LevelData<Data> >() {}
  LevelField(Field::data_location data_at) :
    GenericField<LevelMesh, LevelData<Data> >(data_at) {}
  LevelField(LevelMeshHandle mesh, Field::data_location data_at) : 
    GenericField<LevelMesh, LevelData<Data> >(mesh, data_at) {}
  
  virtual ~LevelField(){}

  virtual LevelField<Data> *clone() const 
    { return new LevelField<Data>(*this); }
 
  static const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }
  virtual ScalarFieldInterface* query_scalar_interface() const;
  virtual VectorFieldInterface* query_vector_interface() const;
  virtual TensorFieldInterface* query_tensor_interface() const;
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  bool get_gradient(Vector &, const Point &);
  bool interpolate(Data&, const Point&) const;
  bool minmax(pair<double, double>& mm) const ;
  virtual const SCIRun::TypeDescription* get_type_description() const;
private:
  static Persistent* maker();
  class make_minmax_thread : public Runnable
    {
    public:
      make_minmax_thread( typename Array3<Data>::iterator it,
			  typename Array3<Data>::iterator it_end,
			  double& min, double& max, Semaphore* sema,
			  Mutex& m):
	it_(it), it_end_(it_end), min_(min), max_(max), sema_(sema), m_(m){}

      void run()
	{
	  double min, max;
	  min = max = *it_;
	  ++it_;
	  for(; it_ != it_end_; ++it_){
	    min = Min(min, double(*it_));
	    max = Max(max, double(*it_));
	  }
	  m_.lock();
	  min_ = Min(min_, min);
	  max_ = Max(max_, max);
	  m_.unlock();
	  sema_->up();
	}
    private:
      typename Array3<Data>::iterator it_;
      typename Array3<Data>::iterator it_end_;
      double &min_, &max_;
      Semaphore *sema_;
      Mutex& m_;
    };
};

template<>
void LevelField<Matrix3>::make_minmax_thread::run();
template<>
void LevelField<Vector>::make_minmax_thread::run();

#define LEVELFIELD_VERSION 1

template <class Data>
Persistent* 
LevelField<Data>::maker()
{
  return scinew LevelField<Data>;
}

template <class Data>
PersistentTypeID
LevelField<Data>::type_id(type_name(-1),
		GenericField<LevelMesh, LevelData<Data> >::type_name(-1),
                maker); 

template <class Data>
void
LevelField<Data>::io(Piostream &stream)
{
  stream.begin_class(type_name(-1), LEVELFIELD_VERSION);
  GenericField<LevelMesh, LevelData<Data> >::io(stream);
  stream.end_class();                                                         
}

//! Virtual interface.
template <class Data>
ScalarFieldInterface *
LevelField<Data>::query_scalar_interface() const
{
  return 0;
}

template <> ScalarFieldInterface *
LevelField<double>::query_scalar_interface() const;

template <> ScalarFieldInterface *
LevelField<float>::query_scalar_interface() const;

template <> ScalarFieldInterface *
LevelField<long>::query_scalar_interface() const;



template < class Data>
VectorFieldInterface* 
LevelField<Data>::query_vector_interface() const
{
  return 0;
}

template <>
VectorFieldInterface*
LevelField<Vector>::query_vector_interface() const ;




template <class Data>
TensorFieldInterface* 
LevelField<Data>::query_tensor_interface() const
{
  ASSERTFAIL("LevelField::query_tensor_interface() not implemented");
}

template <class Data>
const string
LevelField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "LevelField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 


template <class T>
const SCIRun::TypeDescription* 
LevelField<T>::get_type_description() const 
{
  return SCIRun::get_type_description((LevelField<T>*)0);
}




//! compute the gradient g, at point p
template <> 
bool LevelField<Matrix3>::get_gradient(Vector &, const Point &p);

template <> 
bool LevelField<Vector>::get_gradient(Vector &, const Point &p);

template <class Data>
bool LevelField<Data>::get_gradient(Vector &g, const Point &p) {
  // for now we only know how to do this for fields with scalars at the nodes

  if( type_name(1) == "double" ||
      type_name(1) == "long" ||
      type_name(1) == "int" ) {

    if( data_at() == Field::NODE){
      mesh_handle_type mesh = get_typed_mesh();
      Vector pn=p-mesh->get_min();
      Vector diagonal = mesh->diagonal();
      int nx=mesh->get_nx();
      int ny=mesh->get_ny();
      int nz=mesh->get_nz();
      double diagx=diagonal.x();
      double diagy=diagonal.y();
      double diagz=diagonal.z();
      double x=pn.x()*(nx-1)/diagx;
      double y=pn.y()*(ny-1)/diagy;
      double z=pn.z()*(nz-1)/diagz;
      int ix0=(int)x;
      int iy0=(int)y;
      int iz0=(int)z;
      int ix1=ix0+1;
      int iy1=iy0+1;
      int iz1=iz0+1;
      if(ix0<0 || ix1>=nx)return false;
      if(iy0<0 || iy1>=ny)return false;
      if(iz0<0 || iz1>=nz)return false;
      double fx=x-ix0;
      double fy=y-iy0;
      double fz=z-iz0;
      double d000=(double)value(mesh->node(ix0,iy0,iz0));
      double d100=(double)value(mesh->node(ix1,iy0,iz0));
      double d010=(double)value(mesh->node(ix0,iy1,iz0));
      double d110=(double)value(mesh->node(ix1,iy1,iz0));
      double d001=(double)value(mesh->node(ix0,iy0,iz1));
      double d101=(double)value(mesh->node(ix1,iy0,iz1));
      double d011=(double)value(mesh->node(ix0,iy1,iz1));
      double d111=(double)value(mesh->node(ix1,iy1,iz1));
      double z00=Interpolate(d000, d001, fz);
      double z01=Interpolate(d010, d011, fz);
      double z10=Interpolate(d100, d101, fz);
      double z11=Interpolate(d110, d111, fz);
      double yy0=Interpolate(z00, z01, fy);
      double yy1=Interpolate(z10, z11, fy);
      double dx=(yy1-yy0)*(nx-1)/diagx;
      double x00=Interpolate(d000, d100, fx);
      double x01=Interpolate(d001, d101, fx);
      double x10=Interpolate(d010, d110, fx);
      double x11=Interpolate(d011, d111, fx);
      double y0=Interpolate(x00, x10, fy);
      double y1=Interpolate(x01, x11, fy);
      double dz=(y1-y0)*(nz-1)/diagz;
      double z0=Interpolate(x00, x01, fz);
      double z1=Interpolate(x10, x11, fz);
      double dy=(z1-z0)*(ny-1)/diagy;
      g = Vector(dx, dy, dz);
      return true;
    }
  }
  return false;
}

template <class Data>
bool LevelField<Data>::interpolate(Data &g, const Point &p) const {
  // for now we only know how to do this for fields with scalars at the nodes
  mesh_handle_type mesh = get_typed_mesh();
  int nx, ny, nz;
  double x,y,z;
  Vector pn=p-mesh->get_min();
  Vector diagonal = mesh->diagonal();
  if(data_at() == Field::NODE) {
    nx=mesh->get_nx();
    ny=mesh->get_ny();
    nz=mesh->get_nz();
    x=pn.x()*(nx-1)/diagonal.x();
    y=pn.y()*(ny-1)/diagonal.y();
    z=pn.z()*(nz-1)/diagonal.z();
    int ix0=(int)x;
    int iy0=(int)y;
    int iz0=(int)z;
    int ix1=ix0+1;
    int iy1=iy0+1;
    int iz1=iz0+1;
    if(ix0<0 || ix1>=nx)return false;
    if(iy0<0 || iy1>=ny)return false;
    if(iz0<0 || iz1>=nz)return false;
    double fx=x-ix0;
    double fy=y-iy0;
    double fz=z-iz0;
    Data x00=Interpolate(value(mesh->node(ix0,iy0,iz0)),
			 value(mesh->node(ix1,iy0,iz0)), fx);
    Data x01=Interpolate(value(mesh->node(ix0,iy0,iz1)),
			 value(mesh->node(ix1,iy0,iz1)), fx);
    Data x10=Interpolate(value(mesh->node(ix0,iy1,iz0)),
			 value(mesh->node(ix1,iy1,iz1)), fx);
    Data x11=Interpolate(value(mesh->node(ix0,iy1,iz1)),
			 value(mesh->node(ix1,iy1,iz1)), fx);
    Data y0=Interpolate(x00, x10, fy);
    Data y1=Interpolate(x01, x11, fy);
    g=Interpolate(y0, y1, fz);
  } else if( data_at() == Field::CELL) {
    typename mesh_type::Cell::index_type ci;
    if( mesh->locate(ci, p) ) {
      g = value( ci );
    } else {
      return false;
    }
/*     nx=mesh->get_nx()-1; */
/*     ny=mesh->get_ny()-1; */
/*     nz=mesh->get_nz()-1; */
/*     x=pn.x()*(nx-1)/diagonal.x(); */
/*     y=pn.y()*(ny-1)/diagonal.y(); */
/*     z=pn.z()*(nz-1)/diagonal.z(); */
/*     int ix0=(int)(x); */
/*     int iy0=(int)(y); */
/*     int iz0=(int)(z); */
/*     if(ix0<0 || ix0>=nx-1)return false; */
/*     if(iy0<0 || iy0>=ny-1)return false; */
/*     if(iz0<0 || iz0>=nz-1)return false; */
    
/*     g = value(mesh->cell(ix0,iy0,iz0)); */
    
  } else {
    return false;
  }
  return true;
}

template <>
bool LevelField<Vector>::minmax( pair<double, double> & mm) const;

template <>
bool LevelField<Matrix3>::minmax( pair<double, double> & mm) const;

template <class Data>
bool LevelField<Data>::minmax( pair<double, double> & mm) const
{
  if(type_name(1) == "double" || type_name(1) == "int" ||
     type_name(1) == "long"){
    double mn, mx;
    fdata_type dt = fdata();
    vector<ShareAssignArray3<Data> > vdt = fdata();
    vector<ShareAssignArray3<Data> >::iterator vit = vdt.begin();
    vector<ShareAssignArray3<Data> >::iterator vit_end = vdt.end();
    int max_workers = Max(Thread::numProcessors()/3, 4);

    Semaphore* thread_sema = scinew Semaphore( "scalar extractor semaphore",
					       max_workers); 
    Mutex lock("make_minmax_thread mutex");
    //    typename fdata_type::iterator it = dt.begin();
    //    typename fdata_type::iterator it_end = dt.end();
    mn = mx = *((*vit).begin());
    for(;vit != vit_end; ++vit) {
      Array3<Data>::iterator it((*vit).begin());
      Array3<Data>::iterator it_end((*vit).end());
      
      thread_sema->down();
      Thread *thrd = 
	scinew Thread(scinew LevelField<Data>::make_minmax_thread( it, it_end,
						   mn, mx, thread_sema, lock),
		      "minmax worker" );
      thrd->detach();
    }
    thread_sema->down( max_workers );
    if( thread_sema ) delete thread_sema;
    mm.first = mn;
    mm.second = mx;
    return true;
  } else { 
    return false;
  }
}
template <>
bool LevelFieldSFI<double>::interpolate( double& result, const Point &p) const;

template <>
bool LevelFieldSFI<float>::interpolate( double& result, const Point &p) const;

template <>
bool LevelFieldSFI<long>::interpolate( double& result, const Point &p) const;
  
template <class Data>
bool LevelFieldSFI<Data>::compute_min_max(double &minout, double& maxout) const
{
  pair<double, double> tmp;
  bool result = fld_->minmax( tmp );
  minout = tmp.first;
  maxout = tmp.second;
  return result;
}

template <class Data>
bool LevelFieldSFI<Data>::interpolate( double& result, const Point &p) const
{
  return false;
}
template <class Data> 
bool LevelFieldSFI<Data>::interpolate_many(vector<double> &results,
				const vector<Point> &points) const
{
  return false;
}

} // namespace Uintah

namespace SCIRun {
#define LEVELDATA_VERSION 1

template<class T>
void Pio(Piostream& stream, Uintah::LevelData<T>& data)
{
#ifdef __GNUG__
#else
#endif

  stream.begin_class("Level", LEVELDATA_VERSION);
  NOT_FINISHED("Uintah::LevelData::io");

  stream.end_class();
}


template <class T>
const TypeDescription* 
get_type_description(Uintah::LevelField<T>*)
{
  static TypeDescription* td = 0;
  static string name("LevelField");
  static string namesp("Uintah");
  static string path(__FILE__);
  if(!td){
    const TypeDescription *sub = get_type_description((T*)0);
    TypeDescription::td_vec *subs =
      scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription(name, subs, path, namesp);
  }
  return td;
}

} // end namespace SCIRun

#endif // Datatypes_LevelField_h
