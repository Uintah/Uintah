/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


#ifndef CORE_ALGORITHMS_ARRAYOBJECTFIELDALGO_H
#define CORE_ALGORITHMS_ARRAYOBJECTFIELDALGO_H 1

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

// Basis classes
#include <Core/Basis/NoData.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Basis/CrvQuadraticLgn.h>
#include <Core/Basis/HexTricubicHmt.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Basis/HexTriquadraticLgn.h>
#include <Core/Basis/PrismCubicHmt.h>
#include <Core/Basis/PrismLinearLgn.h>
#include <Core/Basis/PrismQuadraticLgn.h>
#include <Core/Basis/QuadBicubicHmt.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/QuadBiquadraticLgn.h>
#include <Core/Basis/TetCubicHmt.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Basis/TetQuadraticLgn.h>
#include <Core/Basis/TriCubicHmt.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Basis/TriQuadraticLgn.h>

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/ImageMesh.h>
#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/StructQuadSurfMesh.h>
#include <Core/Containers/FData.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Containers/HashTable.h>
#include <Core/Datatypes/Field.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>

#include <Core/Algorithms/ArrayMath/ArrayEngineMath.h>
#include <Core/Algorithms/ArrayMath/share.h>

namespace SCIRunAlgo {

using namespace SCIRun;

////////// ArrayObjectFieldDataAlgo //////////////////

class SCISHARE ArrayObjectFieldDataAlgo : public SCIRun::DynamicAlgoBase {
  public:
    virtual void getnextscalar(DataArrayMath::Scalar& scalar);
    virtual void getnextvector(DataArrayMath::Vector& vector);
    virtual void getnexttensor(DataArrayMath::Tensor& tensor);

    virtual void setnextscalar(DataArrayMath::Scalar& scalar);
    virtual void setnextvector(DataArrayMath::Vector& vector);
    virtual void setnexttensor(DataArrayMath::Tensor& tensor);

    virtual bool isscalar();
    virtual bool isvector();
    virtual bool istensor();
    virtual int  size();

    virtual bool setfield(SCIRun::FieldHandle handle);
    virtual void reset();
    virtual void reset(unsigned int offset);
    
    static  SCIRun::CompileInfoHandle get_compile_info(SCIRun::FieldHandle& field);
};

////////// ArrayObjectFieldDataScalarAlgoT //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldDataScalarAlgoT : public ArrayObjectFieldDataAlgo {
  public:
    virtual void getnextscalar(DataArrayMath::Scalar& scalar);
    virtual void setnextscalar(DataArrayMath::Scalar& scalar);
    virtual bool isscalar();
    virtual int  size();

    virtual bool  setfield(SCIRun::FieldHandle handle);    
    virtual void  reset();
    virtual void reset(unsigned int offset);
    
  private:
    typename LOC::iterator it_;
    FIELD*  field_;
    SCIRun::FieldHandle handle_;
};

template<class FIELD, class LOC>
void ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::reset()
{
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->begin(it_);
  for (unsigned int p=0;p<offset;p++) ++it_;
}



template<class FIELD, class LOC>
bool ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;
  field_ = dynamic_cast<FIELD*>(handle.get_rep());
  if (field_ == 0) return(false);
  
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  if (mesh == 0) return(false);
  mesh->begin(it_);
  
  return (true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::getnextscalar(DataArrayMath::Scalar& scalar)
{
  typename FIELD::value_type t;
  if (field_->basis_order() == -1) { scalar = 0.0; return; }
  field_->value(t,(*it_));
  scalar = static_cast<double>(t);
  ++it_;  
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::setnextscalar(DataArrayMath::Scalar& scalar)
{
  if (field_->basis_order() == -1) { return; }
  typename FIELD::value_type t = static_cast<typename FIELD::value_type>(scalar);
  field_->set_value(t,(*it_));
  ++it_;  
}

template<class FIELD, class LOC>
bool ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::isscalar()
{
  return(true);
}

template<class FIELD, class LOC>
int ArrayObjectFieldDataScalarAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->size(s);
  return(static_cast<int>(s));
}

////////// ArrayObjectFieldVectorAlgo //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldDataVectorAlgoT : public ArrayObjectFieldDataAlgo {
  public:
    virtual void getnextvector(DataArrayMath::Vector& vector);
    virtual void setnextvector(DataArrayMath::Vector& vector);
    virtual bool isvector();
    virtual int  size();

    virtual bool setfield(SCIRun::FieldHandle handle);    
    virtual void reset();
    virtual void reset(unsigned int offset);
    
  private:
    typename LOC::iterator it_;
    FIELD*  field_;
    SCIRun::FieldHandle handle_;
};

template<class FIELD, class LOC>
void ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::reset()
{
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->begin(it_);
  for (unsigned int p=0;p<offset;p++) ++it_;
}


template<class FIELD, class LOC>
bool ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;
  field_ = dynamic_cast<FIELD*>(handle.get_rep());
  if (field_ == 0) return(false);
  
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  if (mesh == 0) return(false);
  
  mesh->begin(it_);
  return(true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::getnextvector(DataArrayMath::Vector& vector)
{
  SCIRun::Vector v;
  if (field_->basis_order() == -1) { vector = 0.0; return; }
  field_->value(v,(*it_));
  vector = DataArrayMath::Vector(v.x(),v.y(),v.z());
  ++it_;  
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::setnextvector(DataArrayMath::Vector& vector)
{
  if (field_->basis_order() == -1) { return; }
  SCIRun::Vector v(vector.x(),vector.y(),vector.z());
  field_->set_value(v,(*it_));
  ++it_;  
}

template<class FIELD, class LOC>
bool ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::isvector()
{
  return(true);
}

template<class FIELD, class LOC>
int ArrayObjectFieldDataVectorAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->size(s);
  return(static_cast<int>(s));
}

////////// ArrayObjectFieldDataTensorAlgo //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldDataTensorAlgoT : public ArrayObjectFieldDataAlgo {
  public:
    virtual void getnexttensor(DataArrayMath::Tensor& tensor);
    virtual void setnexttensor(DataArrayMath::Tensor& tensor);
    virtual bool istensor();
    virtual int  size();

    virtual bool setfield(SCIRun::FieldHandle handle);    
    virtual void reset();
    virtual void reset(unsigned int);
    
  private:
    typename LOC::iterator it_;
    FIELD*  field_;
    SCIRun::FieldHandle handle_;
};

template<class FIELD, class LOC>
void ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::reset()
{
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->begin(it_);
  for (int p=0; p< offset; p++) ++it_;
}

template<class FIELD, class LOC>
bool ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;
  field_ = dynamic_cast<FIELD*>(handle.get_rep());
  if (field_ == 0) return(false);
  
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep());
  if (mesh == 0) return(false);
   
  mesh->begin(it_);
  return(true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::getnexttensor(DataArrayMath::Tensor& tensor)
{
  // The SCIRun Tensor class is very limited and badly designed, hence we use a newly
  // designed class with more possiblities, we just convert between both.

  if (field_->basis_order() == -1) { tensor = 0.0; return; }

  SCIRun::Tensor t;
  field_->value(t,(*it_));
  tensor = DataArrayMath::Tensor(t.mat_[0][0],t.mat_[1][0],t.mat_[2][0],t.mat_[1][1],t.mat_[2][1],t.mat_[2][2]);
  ++it_;  
}

template<class FIELD, class LOC>
void ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::setnexttensor(DataArrayMath::Tensor& tensor)
{
  if (field_->basis_order() == -1) { return; }
  field_->set_value(SCIRun::Tensor(tensor.getdataptr()),(*it_));
  ++it_;  
}

template<class FIELD, class LOC>
bool ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::istensor()
{
  return(true);
}

template<class FIELD, class LOC>
int ArrayObjectFieldDataTensorAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(field_->mesh().get_rep()); 
  mesh->size(s);
  return(static_cast<int>(s));
}

////////////////////////////////////////////////////////////////////////////////

////////// ArrayObjectFieldLocationAlgo //////////////////

class SCISHARE ArrayObjectFieldLocationAlgo : public SCIRun::DynamicAlgoBase {
  public:
    virtual void getnextlocation(DataArrayMath::Vector& vector) = 0;
    virtual void setnextlocation(DataArrayMath::Vector& vector) = 0;
 
    virtual int  size() = 0;
    virtual bool setfield(SCIRun::FieldHandle handle) = 0;
    virtual void reset() = 0;
    virtual void reset(unsigned int offset) = 0;
    
    static  SCIRun::CompileInfoHandle get_compile_info(SCIRun::FieldHandle& field);
};

////////// ArrayObjectFSetieldLocationAlgo //////////////////

class SCISHARE ArrayObjectSetFieldLocationAlgo : public SCIRun::DynamicAlgoBase {
  public:
    virtual void setnextlocation(DataArrayMath::Vector& vector) = 0;
    virtual int  size() = 0;
    virtual bool setfield(SCIRun::FieldHandle handle) = 0;
    virtual void reset() = 0;
    virtual void reset(unsigned int offset) = 0;
    
    static  SCIRun::CompileInfoHandle get_compile_info(SCIRun::FieldHandle& field);
};


////////// ArrayObjectFieldLocationNodeAlgoT //////////////////

template<class FIELD>
class SCISHARE ArrayObjectFieldLocationAlgoT : public ArrayObjectFieldLocationAlgo {
  public:
    virtual void getnextlocation(DataArrayMath::Vector& vector);
    virtual void setnextlocation(DataArrayMath::Vector& vector);
   
    virtual int  size();
    virtual bool setfield(SCIRun::FieldHandle handle);    
    virtual void reset();
    virtual void reset(unsigned int offset);
    
  private:
    typename FIELD::mesh_type::Node::iterator it_;
    typename FIELD::mesh_handle_type mesh_;
    FIELD*  field_;
    SCIRun::FieldHandle handle_;
    SCIRun::MeshHandle meshhandle_;
};

template<class FIELD>
void ArrayObjectFieldLocationAlgoT<FIELD>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD>
void ArrayObjectFieldLocationAlgoT<FIELD>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned int p=0;p<offset;p++) ++it_;
}


template<class FIELD>
bool ArrayObjectFieldLocationAlgoT<FIELD>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;
  field_ = dynamic_cast<FIELD*>(handle.get_rep());
  if (field_ == 0) return(false);
  
  mesh_ = field_->get_typed_mesh();
  mesh_->begin(it_);
  return(true);
}

template<class FIELD>
void ArrayObjectFieldLocationAlgoT<FIELD>::getnextlocation(DataArrayMath::Vector& location)
{
  SCIRun::Point p;
  mesh_->get_point(p,(*it_));
  location = DataArrayMath::Vector(p.x(),p.y(),p.z());
  ++it_;  
}

template<class FIELD>
void ArrayObjectFieldLocationAlgoT<FIELD>::setnextlocation(DataArrayMath::Vector& location)
{
  mesh_->set_point(SCIRun::Point(location.x(),location.y(),location.z()),(*it_));
  ++it_;
}

template<class FIELD>
int ArrayObjectFieldLocationAlgoT<FIELD>::size()
{
  typename FIELD::mesh_type::Node::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}



////////// ArrayObjectFieldLocationNodeAlgoT //////////////////

template<class FIELD>
class SCISHARE ArrayObjectFieldLocationNodeAlgoT : public ArrayObjectFieldLocationAlgo {
  public:
    virtual void getnextlocation(DataArrayMath::Vector& vector);
    virtual void setnextlocation(DataArrayMath::Vector& vector);
   
    virtual int  size();
    virtual bool setfield(SCIRun::FieldHandle handle);    
    virtual void reset();
    virtual void reset(unsigned int offset);
    
  private:
    typename FIELD::mesh_type::Node::iterator it_;
    typename FIELD::mesh_handle_type mesh_;
    FIELD*  field_;
    SCIRun::FieldHandle handle_;
    SCIRun::MeshHandle meshhandle_;
};

template<class FIELD>
void ArrayObjectFieldLocationNodeAlgoT<FIELD>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD>
void ArrayObjectFieldLocationNodeAlgoT<FIELD>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned int p=0; p< offset;p++) ++it_;
}

template<class FIELD>
bool ArrayObjectFieldLocationNodeAlgoT<FIELD>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;
  field_ = dynamic_cast<FIELD*>(handle.get_rep());
  if (field_ == 0) return(false);
  
  mesh_ = field_->get_typed_mesh();
  mesh_->begin(it_);
  return(true);
}

template<class FIELD>
void ArrayObjectFieldLocationNodeAlgoT<FIELD>::getnextlocation(DataArrayMath::Vector& location)
{
  SCIRun::Point p;
  mesh_->get_point(p,(*it_));
  location = DataArrayMath::Vector(p.x(),p.y(),p.z());
  ++it_;  
}

template<class FIELD>
void ArrayObjectFieldLocationNodeAlgoT<FIELD>::setnextlocation(DataArrayMath::Vector& location)
{
}

template<class FIELD>
int ArrayObjectFieldLocationNodeAlgoT<FIELD>::size()
{
  typename FIELD::mesh_type::Node::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}


////////// ArrayObjectFieldLocationElemAlgoT //////////////////

template<class FIELD>
class SCISHARE ArrayObjectFieldLocationElemAlgoT : public ArrayObjectFieldLocationAlgo {
  public:
    virtual void getnextlocation(DataArrayMath::Vector& vector);
    virtual void setnextlocation(DataArrayMath::Vector& vector);
    virtual int  size();
    virtual bool setfield(SCIRun::FieldHandle handle);    
    virtual void reset();
    virtual void reset(unsigned int offset);
    
  private:
    typename FIELD::mesh_type::Elem::iterator it_;
    typename FIELD::mesh_handle_type mesh_; 
    FIELD*  field_;
    SCIRun::FieldHandle handle_;
};

template<class FIELD>
void ArrayObjectFieldLocationElemAlgoT<FIELD>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD>
void ArrayObjectFieldLocationElemAlgoT<FIELD>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned int p=0; p<offset; p++) ++it_;
}


template<class FIELD>
bool ArrayObjectFieldLocationElemAlgoT<FIELD>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;
  field_ = dynamic_cast<FIELD*>(handle.get_rep());
  if (field_ == 0) return(false);
  mesh_ = field_->get_typed_mesh(); 
  mesh_->begin(it_);
  return(true);
}

template<class FIELD>
void ArrayObjectFieldLocationElemAlgoT<FIELD>::getnextlocation(DataArrayMath::Vector& location)
{
  SCIRun::Point p;
  mesh_->get_center(p,(*it_));
  location = DataArrayMath::Vector(p.x(),p.y(),p.z());
  ++it_;  
}

template<class FIELD>
void ArrayObjectFieldLocationElemAlgoT<FIELD>::setnextlocation(DataArrayMath::Vector& location)
{
}


template<class FIELD>
int ArrayObjectFieldLocationElemAlgoT<FIELD>::size()
{
  typename FIELD::mesh_type::Elem::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}

////////// ArrayObjectFieldCreateAlgo //////////////////

class SCISHARE ArrayObjectFieldCreateAlgo : public SCIRun::DynamicAlgoBase {
  public:
    virtual bool createfield(SCIRun::FieldHandle input,SCIRun::FieldHandle& output) = 0;
    static  SCIRun::CompileInfoHandle get_compile_info(SCIRun::FieldHandle field,std::string datatype, std::string basistype = "");
};

template<class FIELD>
class ArrayObjectFieldCreateAlgoT : public ArrayObjectFieldCreateAlgo {
  public:
    virtual bool createfield(SCIRun::FieldHandle input,SCIRun::FieldHandle& output);
};

template<class FIELD>
bool ArrayObjectFieldCreateAlgoT<FIELD>::createfield(SCIRun::FieldHandle input,SCIRun::FieldHandle& output)
{
  
  typename FIELD::mesh_type* mesh = dynamic_cast<typename FIELD::mesh_type *>(input->mesh().get_rep());
  if (mesh == 0) return(false);
  FIELD* ofield = scinew FIELD(mesh);
  output = dynamic_cast<SCIRun::Field *>(ofield);
  if (output.get_rep() == 0) return(false);
  ofield->resize_fdata();
  output->copy_properties(input.get_rep()); 
  return(true);
}


////////// ArrayObjectFieldElemAlgo //////////////////

class SCISHARE ArrayObjectFieldElemAlgo : public SCIRun::DynamicAlgoBase {
  public:
    virtual void getcenter(DataArrayMath::Vector& node);
    virtual void getsize(DataArrayMath::Scalar& size);
    virtual void getlength(DataArrayMath::Scalar& length);
    virtual void getarea(DataArrayMath::Scalar& area);
    virtual void getvolume(DataArrayMath::Scalar& volume);
    virtual void getdimension(DataArrayMath::Scalar& dim);
    virtual void getnormal(DataArrayMath::Vector& normal);
    
    virtual bool ispoint();
    virtual bool isline();
    virtual bool issurface();
    virtual bool isvolume();

    virtual bool setfield(SCIRun::FieldHandle handle);
    virtual void reset();
    virtual void reset(unsigned int offset);
    virtual void next();
    virtual int  size();    
    
    static SCIRun::CompileInfoHandle get_compile_info(SCIRun::FieldHandle field);

    void get_normal(SCIRun::TriSurfMesh<TriLinearLgn<Point> > *mesh,SCIRun::TriSurfMesh<TriLinearLgn<Point> >::Face::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> >::Face::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::TriSurfMesh<TriLinearLgn<Point> > *mesh,SCIRun::TriSurfMesh<TriLinearLgn<Point> >::Node::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::QuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::ImageMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::ImageMesh<QuadBilinearLgn<Point> >::Face::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::ImageMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::ImageMesh<QuadBilinearLgn<Point> >::Node::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> >::Face::iterator& it,DataArrayMath::Vector& vec);
    void get_normal(SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> > *mesh,SCIRun::StructQuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator& it,DataArrayMath::Vector& vec);


};

////////// ArrayObjectFieldElemPointAlgo //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldElemPointAlgoT : public ArrayObjectFieldElemAlgo {
  public:
    virtual void getcenter(DataArrayMath::Vector& node);
    virtual void getdimension(DataArrayMath::Scalar& dim);

    virtual bool setfield(SCIRun::FieldHandle handle);
    virtual void reset();
    virtual void reset(unsigned int offset);

    virtual void next(); 
    virtual int  size();       
  
  private:
    typename LOC::iterator it_;
    typename FIELD::mesh_type* mesh_;
    SCIRun::FieldHandle handle_;    
};


template<class FIELD, class LOC>
int ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}

template<class FIELD, class LOC>
bool ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;

  FIELD* field = dynamic_cast<FIELD*>(handle.get_rep());
  if (field == 0) return(false);

  mesh_ = dynamic_cast<typename FIELD::mesh_type *>(handle->mesh().get_rep());
  if (mesh_ == 0) return(false);

  mesh_->begin(it_);
  return(true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned int p=0;p<offset;p++) ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::next()
{
  ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::getcenter(DataArrayMath::Vector& node)
{
  SCIRun::Point p;
  mesh_->get_center(p,*it_);
  node =DataArrayMath:: Vector(p.x(),p.y(),p.z());
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemPointAlgoT<FIELD,LOC>::getdimension(DataArrayMath::Scalar& dim)
{
  dim = 0.0;
}

////////// ArrayObjectFieldElemLineAlgo //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldElemLineAlgoT : public ArrayObjectFieldElemAlgo {
  public:
    virtual void getcenter(DataArrayMath::Vector& node);
    virtual void getsize(DataArrayMath::Scalar& size);
    virtual void getlength(DataArrayMath::Scalar& length);
    virtual void getdimension(DataArrayMath::Scalar& dim);

    virtual bool setfield(SCIRun::FieldHandle handle);
    virtual void reset();
    virtual void reset(unsigned int offset);
    virtual void next(); 
    virtual int  size();       
  
  private:
    typename LOC::iterator it_;
    typename FIELD::mesh_type* mesh_;
    SCIRun::FieldHandle handle_;    
};

template<class FIELD, class LOC>
int ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}

template<class FIELD, class LOC>
bool ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;

  FIELD* field = dynamic_cast<FIELD*>(handle.get_rep());
  if (field == 0) return(false);

  mesh_ = dynamic_cast<typename FIELD::mesh_type *>(handle->mesh().get_rep());
  if (mesh_ == 0) return(false);

  mesh_->begin(it_);
  return(true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned int p = 0; p<offset; p++) ++it_;
}


template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::next()
{
  ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::getcenter(DataArrayMath::Vector& node)
{
  SCIRun::Point p;
  mesh_->get_center(p,*it_);
  node = DataArrayMath::Vector(p.x(),p.y(),p.z());
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::getdimension(DataArrayMath::Scalar& dim)
{
  dim = 1.0;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::getlength(DataArrayMath::Scalar& length)
{
  length = mesh_->get_length(*it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemLineAlgoT<FIELD,LOC>::getsize(DataArrayMath::Scalar& size)
{
  size = mesh_->get_size(*it_);
}

////////// ArrayObjectFieldElemSurfAlgo //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldElemSurfAlgoT : public ArrayObjectFieldElemAlgo {
  public:
    virtual void getcenter(DataArrayMath::Vector& node);
    virtual void getsize(DataArrayMath::Scalar& size);
    virtual void getlength(DataArrayMath::Scalar& length);
    virtual void getarea(DataArrayMath::Scalar& area);
    virtual void getdimension(DataArrayMath::Scalar& dim);
    virtual void getnormal(DataArrayMath::Vector& normal);

    virtual bool setfield(SCIRun::FieldHandle handle);
    virtual void reset();
    virtual void reset(unsigned int offset);
    virtual void next();    
    virtual int  size();    
  
  private:
    typename LOC::iterator it_;
    typename FIELD::mesh_type* mesh_;
    SCIRun::FieldHandle handle_;    
};

template<class FIELD, class LOC>
int ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}

template<class FIELD, class LOC>
bool ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;

  FIELD* field = dynamic_cast<FIELD*>(handle.get_rep());
  if (field == 0) return(false);

  mesh_ = dynamic_cast<typename FIELD::mesh_type *>(handle->mesh().get_rep());
  if (mesh_ == 0) return(false);

  mesh_->begin(it_);
  return(true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned int p = 0; p < offset; p++) ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::next()
{
  ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::getcenter(DataArrayMath::Vector& node)
{
  SCIRun::Point p;
  mesh_->get_center(p,*it_);
  node = DataArrayMath::Vector(p.x(),p.y(),p.z());
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::getdimension(DataArrayMath::Scalar& dim)
{
  dim = 1.0;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::getlength(DataArrayMath::Scalar& length)
{
  typename FIELD::mesh_type::Edge::array_type a;
  mesh_->synchronize(SCIRun::Mesh::EDGES_E);
  mesh_->get_edges(a, *it_);
  length = 0.0;
  for (size_t p=0;p<a.size();p++)
  {
    length += mesh_->get_size(a[p]);
  }
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::getarea(DataArrayMath::Scalar& area)
{
  area = mesh_->get_area(*it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::getsize(DataArrayMath::Scalar& size)
{
  size = mesh_->get_size(*it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemSurfAlgoT<FIELD,LOC>::getnormal(DataArrayMath::Vector& normal)
{
  ArrayObjectFieldElemAlgo::get_normal(mesh_,it_,normal);
}



////////// ArrayObjectFieldElemVolumeAlgo //////////////////

template<class FIELD, class LOC>
class SCISHARE ArrayObjectFieldElemVolumeAlgoT : public ArrayObjectFieldElemAlgo {
  public:
    virtual void getcenter(DataArrayMath::Vector& node);
    virtual void getsize(DataArrayMath::Scalar& size);
    virtual void getlength(DataArrayMath::Scalar& length);
    virtual void getarea(DataArrayMath::Scalar& area);
    virtual void getvolume(DataArrayMath::Scalar& area);
    virtual void getdimension(DataArrayMath::Scalar& dim);

    virtual bool setfield(SCIRun::FieldHandle handle);
    virtual void reset();
    virtual void reset(unsigned int offset);
    virtual void next();    
    virtual int  size();
  
  private:
    typename LOC::iterator it_;
    typename FIELD::mesh_type* mesh_;
    SCIRun::FieldHandle handle_;    
};

template<class FIELD, class LOC>
int ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::size()
{
  typename LOC::size_type s;
  mesh_->size(s);
  return(static_cast<int>(s));
}

template<class FIELD, class LOC>
bool ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::setfield(SCIRun::FieldHandle handle)
{
  handle_ = handle;

  FIELD* field = dynamic_cast<FIELD*>(handle.get_rep());
  if (field == 0) return(false);

  mesh_ = dynamic_cast<typename FIELD::mesh_type *>(handle->mesh().get_rep());
  if (mesh_ == 0) return(false);

  mesh_->begin(it_);
  return(true);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::reset()
{
  mesh_->begin(it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::reset(unsigned int offset)
{
  mesh_->begin(it_);
  for (unsigned p = 0; p < offset; p++) ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::next()
{
  ++it_;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::getcenter(DataArrayMath::Vector& node)
{
  SCIRun::Point p;
  mesh_->get_center(p,*it_);
  node = DataArrayMath::Vector(p.x(),p.y(),p.z());
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::getdimension(DataArrayMath::Scalar& dim)
{
  dim = 3.0;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::getlength(DataArrayMath::Scalar& length)
{
  length = 0.0;
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::getarea(DataArrayMath::Scalar& area)
{
  typename FIELD::mesh_type::Face::array_type a;
  mesh_->synchronize(SCIRun::Mesh::FACES_E);
  mesh_->get_faces(a, *it_);
  area = 0.0;
  for (size_t p=0;p<a.size();p++)
  {
    area += mesh_->get_size(a[p]);
  }
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::getvolume(DataArrayMath::Scalar& volume)
{
  volume = mesh_->get_volume(*it_);
}

template<class FIELD, class LOC>
void ArrayObjectFieldElemVolumeAlgoT<FIELD,LOC>::getsize(DataArrayMath::Scalar& size)
{
  size = mesh_->get_size(*it_);
}

} // end namespace

#endif

