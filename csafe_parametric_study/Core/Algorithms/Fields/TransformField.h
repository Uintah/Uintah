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


#ifndef CORE_ALGORITHMS_FIELDS_TRANSFORMFIELD_H
#define CORE_ALGORITHMS_FIELDS_TRANSFORMFIELD_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Tensor.h>

#include <sci_hash_map.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class TransformFieldAlgo : public DynamicAlgoBase
{
public:
  virtual bool TransformField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata);
};


class TransformFieldScalarAlgo : public TransformFieldAlgo
{
public:
  virtual bool TransformField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata);
};

template <class FIELD>
class TransformFieldVectorAlgoT : public TransformFieldAlgo
{
public:
  virtual bool TransformField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata);
};


template <class FIELD>
bool TransformFieldVectorAlgoT<FIELD>::TransformField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata)
{
  FIELD *ifield = dynamic_cast<FIELD *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("TransformField: Could not obtain input field");
    return (false);
  }

  FIELD *ofield = dynamic_cast<FIELD *>(ifield->clone());
  output = dynamic_cast<Field *>(ofield);
  if (ofield == 0)
  {
    pr->error("TransformField: Could not create output field");    
    return (false);
  }

  ofield->mesh_detach();
  ofield->mesh()->transform(transform);

  if (rotatedata)
  {
    double t[16];
    transform.get(t);
    
    Vector v1(t[0],t[1],t[2]);
    Vector v2(t[4],t[5],t[6]);
    Vector v3(t[8],t[9],t[10]);
    
    v1.normalize();
    v2.normalize();
    v3.normalize();
    
    MatrixHandle R = dynamic_cast<Matrix *>(scinew DenseMatrix(3,3));
    R->put(0,0,v1.x());
    R->put(1,0,v1.y());
    R->put(2,0,v1.z());
    R->put(0,1,v2.x());
    R->put(1,1,v2.y());
    R->put(2,1,v2.z());
    R->put(0,2,v3.x());
    R->put(1,2,v3.y());
    R->put(2,2,v3.z());
    
    MatrixHandle Vi = dynamic_cast<Matrix *>(scinew DenseMatrix(3,1));
    MatrixHandle Vo;
        
    if (ifield->basis_order() == 0)
    {
      Vector ivec,ovec;
      typename FIELD::mesh_type::Elem::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(ivec,*bi);
        Vi->put(0,0,ivec.x());
        Vi->put(1,0,ivec.y());
        Vi->put(2,0,ivec.z());
        Vo = R*Vi;
        ovec.x(Vo->get(0,0));
        ovec.y(Vo->get(1,0));
        ovec.z(Vo->get(2,0));
        ofield->set_value(ovec,*bi);
        ++bi;
      }
    }
    else if (ifield->basis_order() == 1)
    {
      Vector ivec,ovec;
      typename FIELD::mesh_type::Node::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(ivec,*bi);
        Vi->put(0,0,ivec.x());
        Vi->put(1,0,ivec.y());
        Vi->put(2,0,ivec.z());
        Vo = R*Vi;
        ovec.x(Vo->get(0,0));
        ovec.y(Vo->get(1,0));
        ovec.z(Vo->get(2,0));
        ofield->set_value(ovec,*bi);
        ++bi;
      }
    }
    else
    {
      pr->error("TransformField: Rotation has not yet been implemented for non-linear elements");
      return (false);  
    }
  }
  else
  {
    if (ifield->basis_order() == 0)
    {
      Vector vec;
      typename FIELD::mesh_type::Elem::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(vec,*bi);
        ofield->set_value(vec,*bi);
        ++bi;
      }
    }
    else if (ifield->basis_order() == 1)
    {
      Vector vec;
      typename FIELD::mesh_type::Node::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(vec,*bi);
        ofield->set_value(vec,*bi);
        ++bi;
      }
    }
    else
    {
      pr->error("TransformField: Rotation has not yet been implemented for non-linear elements");
      return (false);  
    }  
  }
	output->copy_properties(input.get_rep());
  return (true);
}


template <class FIELD>
class TransformFieldTensorAlgoT : public TransformFieldAlgo
{
public:
  virtual bool TransformField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata);
};


template <class FIELD>
bool TransformFieldTensorAlgoT<FIELD>::TransformField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, Transform& transform, bool rotatedata)
{
  FIELD *ifield = dynamic_cast<FIELD *>(input.get_rep());
  if (ifield == 0)
  {
    pr->error("TransformField: Could not obtain input field");
    return (false);
  }

  FIELD *ofield = dynamic_cast<FIELD *>(ifield->clone());
  output = dynamic_cast<Field *>(ofield);
  if (ofield == 0)
  {
    pr->error("TransformField: Could not create output field");    
    return (false);
  }

  ofield->mesh_detach();
  ofield->mesh()->transform(transform);


  if (rotatedata)
  {
    double t[16];
    transform.get(t);
    
    Vector v1(t[0],t[1],t[2]);
    Vector v2(t[4],t[5],t[6]);
    Vector v3(t[8],t[9],t[10]);
    
    v1.normalize();
    v2.normalize();
    v3.normalize();
    
    MatrixHandle R = dynamic_cast<Matrix *>(scinew DenseMatrix(3,3));
    R->put(0,0,v1.x());
    R->put(1,0,v1.y());
    R->put(2,0,v1.z());
    R->put(0,1,v2.x());
    R->put(1,1,v2.y());
    R->put(2,1,v2.z());
    R->put(0,2,v3.x());
    R->put(1,2,v3.y());
    R->put(2,2,v3.z());
    
    MatrixHandle Rt = dynamic_cast<Matrix *>(R->transpose());
    MatrixHandle Ti = dynamic_cast<Matrix *>(scinew DenseMatrix(3,3));
    MatrixHandle To;
    
    if (ifield->basis_order() == 0)
    {
      Tensor iten,oten;
      typename FIELD::mesh_type::Elem::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(iten,*bi);
        Ti->put(0,0,iten.mat_[0][0]);
        Ti->put(1,0,iten.mat_[1][0]);
        Ti->put(2,0,iten.mat_[2][0]);
        Ti->put(0,1,iten.mat_[0][1]);
        Ti->put(1,1,iten.mat_[1][1]);
        Ti->put(2,1,iten.mat_[2][1]);
        Ti->put(0,2,iten.mat_[0][2]);
        Ti->put(1,2,iten.mat_[1][2]);
        Ti->put(2,2,iten.mat_[2][2]);
        To = R*Ti*Rt;
        oten.mat_[0][0] = To->get(0,0);
        oten.mat_[1][0] = To->get(1,0);
        oten.mat_[2][0] = To->get(2,0);
        oten.mat_[0][1] = To->get(0,1);
        oten.mat_[1][1] = To->get(1,1);
        oten.mat_[2][1] = To->get(2,1);
        oten.mat_[0][2] = To->get(0,2);
        oten.mat_[1][2] = To->get(1,2);
        oten.mat_[2][2] = To->get(2,2);
        ofield->set_value(oten,*bi);
        ++bi;
      }
    }
    else if (ifield->basis_order() == 1)
    {
      Tensor iten,oten;
      typename FIELD::mesh_type::Node::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(iten,*bi);
        Ti->put(0,0,iten.mat_[0][0]);
        Ti->put(1,0,iten.mat_[1][0]);
        Ti->put(2,0,iten.mat_[2][0]);
        Ti->put(0,1,iten.mat_[0][1]);
        Ti->put(1,1,iten.mat_[1][1]);
        Ti->put(2,1,iten.mat_[2][1]);
        Ti->put(0,2,iten.mat_[0][2]);
        Ti->put(1,2,iten.mat_[1][2]);
        Ti->put(2,2,iten.mat_[2][2]);
        To = R*Ti*Rt;
        oten.mat_[0][0] = To->get(0,0);
        oten.mat_[1][0] = To->get(1,0);
        oten.mat_[2][0] = To->get(2,0);
        oten.mat_[0][1] = To->get(0,1);
        oten.mat_[1][1] = To->get(1,1);
        oten.mat_[2][1] = To->get(2,1);
        oten.mat_[0][2] = To->get(0,2);
        oten.mat_[1][2] = To->get(1,2);
        oten.mat_[2][2] = To->get(2,2);
        ofield->set_value(oten,*bi);
        ++bi;
      }
    }
    else
    {
      pr->error("TransformField: Rotation has not yet been implemented for non-linear elements");
      return (false);  
    }
  }
  else
  {
    if (ifield->basis_order() == 0)
    {
      Tensor ten;
      typename FIELD::mesh_type::Elem::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(ten,*bi);
        ofield->set_value(ten,*bi);
        ++bi;
      }
    }
    else if (ifield->basis_order() == 1)
    {
      Tensor ten;
      typename FIELD::mesh_type::Node::iterator bi,ei;
      
      while (bi != ei)
      {
        ifield->value(ten,*bi);
        ofield->set_value(ten,*bi);
        ++bi;
      }
    }
    else
    {
      pr->error("TransformField: Rotation has not yet been implemented for non-linear elements");
      return (false);  
    }  
  }
  
	output->copy_properties(input.get_rep());
  return (true);
}


} // end namespace SCIRunAlgo

#endif 
