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


#ifndef CORE_ALGORITHMS_ARRAYMATH_ARRAYOBJECT_H
#define CORE_ALGORITHMS_ARRAYMATH_ARRAYOBJECT_H 1

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Core/Algorithms/ArrayMath/ArrayEngineMath.h>
#include <Core/Algorithms/ArrayMath/ArrayEngineMathElement.h>
#include <Core/Algorithms/ArrayMath/ArrayObjectFieldAlgo.h>
#include <Core/Algorithms/ArrayMath/share.h>

namespace SCIRunAlgo {

class SCISHARE ArrayObject {

  public:
    inline ArrayObject(SCIRun::ProgressReporter *pr);
          
    bool create_inputdata(SCIRun::FieldHandle field, std::string name);
    bool create_inputdata(SCIRun::MatrixHandle matrix, std::string name);
    bool create_inputindex(std::string name, std::string sizename);
    bool create_inputlocation(SCIRun::FieldHandle field, std::string locname, std::string xname, std::string yname, std::string zname);
    bool create_inputelement(SCIRun::FieldHandle field, std::string name);
    
    bool create_outputdata(SCIRun::FieldHandle& field, std::string datatype, std::string name,SCIRun::FieldHandle& ofield);
    bool create_outputdata(SCIRun::FieldHandle& field, std::string datatype, std::string basistype, std::string name,SCIRun::FieldHandle& ofield);
    bool create_outputdata(int size, std::string datatype, std::string name,SCIRun::MatrixHandle& omatrix);    
    bool create_outputlocation(SCIRun::FieldHandle& field, std::string locname,SCIRun::FieldHandle& ofield);

    
    // This inline code should almost give direct data access and should 
    // reduce the function call overhead. For the field data, we still have
    // to make one virtual function call, hence some performance is lost here,
    // Making it fully dynamically compiled would become really complicated, not
    // impossible, but harder to organize. But who knows if we need more speed
    // we could upgrade the code.
        
    inline void getnextmatrixscalar(TensorVectorMath::Scalar& scalar);
    inline void getnextmatrixvector(TensorVectorMath::Vector& vector);
    inline void getnextmatrixtensor(TensorVectorMath::Tensor& tensor);
    inline void getmatrixscalar(TensorVectorMath::Scalar& scalar);
    inline void getmatrixvector(TensorVectorMath::Vector& vector);
    inline void getmatrixtensor(TensorVectorMath::Tensor& tensor);

    inline void setnextmatrixscalar(TensorVectorMath::Scalar& scalar);
    inline void setnextmatrixvector(TensorVectorMath::Vector& vector);
    inline void setnextmatrixtensor(TensorVectorMath::Tensor& tensor);
    inline void setmatrixscalar(TensorVectorMath::Scalar& scalar);
    inline void setmatrixvector(TensorVectorMath::Vector& vector);
    inline void setmatrixtensor(TensorVectorMath::Tensor& tensor);

    inline void getnextfieldscalar(TensorVectorMath::Scalar& scalar);
    inline void getnextfieldvector(TensorVectorMath::Vector& vector);
    inline void getnextfieldtensor(TensorVectorMath::Tensor& tensor);

    inline void setnextfieldscalar(TensorVectorMath::Scalar& scalar);
    inline void setnextfieldvector(TensorVectorMath::Vector& vector);
    inline void setnextfieldtensor(TensorVectorMath::Tensor& tensor);
    
    inline void getnextfieldlocation(TensorVectorMath::Vector& vector);
    inline void setnextfieldlocation(TensorVectorMath::Vector& vector);

    inline void getelement(TensorVectorMath::Element& elem);

    inline bool ismatrixscalar();
    inline bool ismatrixvector();
    inline bool ismatrixtensor();
    inline bool iscmatrixscalar();
    inline bool iscmatrixvector();
    inline bool iscmatrixtensor();
    inline bool isfieldscalar();
    inline bool isfieldvector();
    inline bool isfieldtensor();
    inline bool islocation();
    inline bool iselement();
    inline bool isindex();

    inline bool ismatrix();
    inline bool isvalid();
    inline bool isarray();
    inline int  size();
    inline void reset();

    inline std::string getname();
    inline std::string getsizename();
    inline std::string getxname();
    inline std::string getyname();
    inline std::string getzname();
    
  private:
    enum Type { MATRIXSCALAR, MATRIXVECTOR, MATRIXTENSOR, FIELDSCALAR,
                FIELDVECTOR, FIELDTENSOR, LOCATION, INDEX, ELEMENT, INVALID };

    // For dynamic compiler
    SCIRun::ProgressReporter *pr_;

    // Object name in function
    std::string name_;
    Type        type_;
    int         size_;
    
    // Storage for handles
    SCIRun::MatrixHandle  matrix_;
    SCIRun::FieldHandle   field_;
                
    // For Fields: dynamically compiled algorithm to get data. 
    // In a sense this defies the dynamically compiled system as it just
    // builds an object through which we can make a virtual call to the data
    // However, the iterators and the design of the field class is far from
    // optimal, hence it should not matter. This class gets us around some
    // of the ugliest code in SCIRun.
    
    // For Field Data
    SCIRun::Handle<ArrayObjectFieldDataAlgo>      fielddataalgo_;
    SCIRun::Handle<ArrayObjectFieldLocationAlgo>  fieldlocationalgo_;
    SCIRun::Handle<ArrayObjectFieldCreateAlgo>    fieldcreatealgo_;
    SCIRun::Handle<ArrayObjectFieldElemAlgo>      fieldelementalgo_;
    
    // For Matrix Data
    double*       data_;
    int           ncols_;
    int           idx_;

    // For location system
    std::string sizename_;
    std::string xname_;
    std::string yname_;
    std::string zname_;
    
    void clear();
    
};

typedef std::vector<ArrayObject> ArrayObjectList;

inline ArrayObject::ArrayObject(SCIRun::ProgressReporter* pr) :
  pr_(pr),
  type_(INVALID),
  data_(0),
  ncols_(0),
  idx_(0)
{
}

inline bool ArrayObject::isvalid()
{
  return(type_ != INVALID);
}

inline bool ArrayObject::ismatrixscalar()
{
  return((type_ == MATRIXSCALAR)&&(size_ > 1));
}

inline bool ArrayObject::ismatrixvector()
{
  return((type_ == MATRIXVECTOR)&&(size_ > 1));
}

inline bool ArrayObject::ismatrixtensor()
{
  return((type_ == MATRIXTENSOR)&&(size_ > 1));
}

inline bool ArrayObject::iscmatrixscalar()
{
  return((type_ == MATRIXSCALAR)&&(size_ == 1));
}

inline bool ArrayObject::iscmatrixvector()
{
  return((type_ == MATRIXVECTOR)&&(size_ == 1));
}

inline bool ArrayObject::iscmatrixtensor()
{
  return((type_ == MATRIXTENSOR)&&(size_ == 1));
}

inline bool ArrayObject::isfieldscalar()
{
  return(type_ == FIELDSCALAR);
}

inline bool ArrayObject::isfieldvector()
{
  return(type_ == FIELDVECTOR);
}

inline bool ArrayObject::isfieldtensor()
{
  return(type_ == FIELDTENSOR);
}

inline bool ArrayObject::islocation()
{
  return(type_ == LOCATION);
}

inline bool ArrayObject::isindex()
{
  return(type_ == INDEX);
}

inline bool ArrayObject::iselement()
{
  return(type_ == ELEMENT);
}

inline bool ArrayObject::ismatrix()
{
  return((type_ == MATRIXSCALAR)||(type_ == MATRIXVECTOR)||(type_ == MATRIXTENSOR));
}

inline int ArrayObject::size()
{
  return(size_);
}

inline bool ArrayObject::isarray()
{
  return((size_ > 1)||(field_.get_rep()));
}

inline std::string ArrayObject::getname()
{
  return(name_);
}

inline std::string ArrayObject::getsizename()
{
  return(sizename_);
}

inline std::string ArrayObject::getxname()
{
  return(xname_);
}
inline std::string ArrayObject::getyname()
{
  return(yname_);
}
inline std::string ArrayObject::getzname()
{
  return(zname_);
}

inline void ArrayObject::reset()
{
  if (matrix_.get_rep() == 0) idx_ = 0;
  if (field_.get_rep() == 0) fielddataalgo_->reset();
}

inline void ArrayObject::getnextmatrixscalar(TensorVectorMath::Scalar& scalar)
{
  scalar = data_[idx_++];
}

inline void ArrayObject::getmatrixscalar(TensorVectorMath::Scalar& scalar)
{
  scalar = data_[0];  
}

inline void ArrayObject::getnextmatrixvector(TensorVectorMath::Vector& vector)
{
  vector = TensorVectorMath::Vector(data_[idx_],data_[idx_+1],data_[idx_+2]); 
  idx_+=3;
}

inline void ArrayObject::getmatrixvector(TensorVectorMath::Vector& vector)
{
  vector = TensorVectorMath::Vector(data_[0],data_[1],data_[2]); 
}

inline void ArrayObject::getnextmatrixtensor(TensorVectorMath::Tensor& tensor)
{
 if (ncols_ == 6)
 {
    tensor = TensorVectorMath::Tensor(data_[idx_],data_[idx_+1],data_[idx_+2],data_[idx_+3],data_[idx_+4],data_[idx_+5]); 
    idx_+=6;  
  }
  else
  {
    tensor = TensorVectorMath::Tensor(data_[idx_],data_[idx_+1],data_[idx_+2],data_[idx_+4],data_[idx_+5],data_[idx_+8]); 
    idx_+=9; 
  } 
}

inline void ArrayObject::getmatrixtensor(TensorVectorMath::Tensor& tensor)
{
 if (ncols_ == 6)
 {
    tensor = TensorVectorMath::Tensor(data_[0],data_[1],data_[2],data_[3],data_[4],data_[5]); 
  }
  else
  {
    tensor = TensorVectorMath::Tensor(data_[0],data_[1],data_[2],data_[4],data_[5],data_[8]); 
  } 
}

inline void ArrayObject::setnextmatrixscalar(TensorVectorMath::Scalar& scalar)
{
  data_[idx_++] = scalar;
}

inline void ArrayObject::setmatrixscalar(TensorVectorMath::Scalar& scalar)
{
  data_[0] = scalar;
}

inline void ArrayObject::setnextmatrixvector(TensorVectorMath::Vector& vector)
{
    data_[idx_++] = vector.x();
    data_[idx_++] = vector.y();
    data_[idx_++] = vector.z();
}

inline void ArrayObject::setmatrixvector(TensorVectorMath::Vector& vector)
{
    data_[0] = vector.x();
    data_[1] = vector.y();
    data_[2] = vector.z();
}

inline void ArrayObject::setnextmatrixtensor(TensorVectorMath::Tensor& tensor)
{
  if (ncols_ == 6)
  {
    data_[idx_++] = tensor.xx();  
    data_[idx_++] = tensor.xy();  
    data_[idx_++] = tensor.xz();  
    data_[idx_++] = tensor.yy();  
    data_[idx_++] = tensor.yz();  
    data_[idx_++] = tensor.zz();  
  }
  else
  {
    data_[idx_++] = tensor.xx();  
    data_[idx_++] = tensor.xy();  
    data_[idx_++] = tensor.xz();  
    data_[idx_++] = tensor.xy();  
    data_[idx_++] = tensor.yy();  
    data_[idx_++] = tensor.yz();
    data_[idx_++] = tensor.xz();  
    data_[idx_++] = tensor.yz();        
    data_[idx_++] = tensor.zz();  
  }  
}

inline void ArrayObject::setmatrixtensor(TensorVectorMath::Tensor& tensor)
{
  if (ncols_ == 6)
  {
    data_[0] = tensor.xx();  
    data_[1] = tensor.xy();  
    data_[2] = tensor.xz();  
    data_[3] = tensor.yy();  
    data_[4] = tensor.yz();  
    data_[5] = tensor.zz();  
  }
  else
  {
    data_[0] = tensor.xx();  
    data_[1] = tensor.xy();  
    data_[2] = tensor.xz();  
    data_[3] = tensor.xy();  
    data_[4] = tensor.yy();  
    data_[5] = tensor.yz();
    data_[6] = tensor.xz();  
    data_[7] = tensor.yz();        
    data_[8] = tensor.zz();  
  }  
}

inline void ArrayObject::getnextfieldscalar(TensorVectorMath::Scalar& scalar)
{
  fielddataalgo_->getnextscalar(scalar);
}

inline void ArrayObject::getnextfieldvector(TensorVectorMath::Vector& vector)
{
  fielddataalgo_->getnextvector(vector);
}

inline void ArrayObject::getnextfieldtensor(TensorVectorMath::Tensor& tensor)
{
  fielddataalgo_->getnexttensor(tensor);
}

inline void ArrayObject::setnextfieldscalar(TensorVectorMath::Scalar& scalar)
{
  fielddataalgo_->setnextscalar(scalar);
}

inline void ArrayObject::setnextfieldvector(TensorVectorMath::Vector& vector)
{
  fielddataalgo_->setnextvector(vector);
}

inline void ArrayObject::setnextfieldtensor(TensorVectorMath::Tensor& tensor)
{
  fielddataalgo_->setnexttensor(tensor);
}

inline void ArrayObject::getnextfieldlocation(TensorVectorMath::Vector& vector)
{
  fieldlocationalgo_->getnextlocation(vector);
}

inline void ArrayObject::setnextfieldlocation(TensorVectorMath::Vector& vector)
{
  fieldlocationalgo_->setnextlocation(vector);
}

inline void ArrayObject::getelement(TensorVectorMath::Element& elem)
{
  elem = TensorVectorMath::Element(fieldelementalgo_);
}

} // end namespace

#endif
