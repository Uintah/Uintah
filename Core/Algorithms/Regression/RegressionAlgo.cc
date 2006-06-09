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

#include <Core/Algorithms/DataIO/DataIOAlgo.h>
#include <Core/Algorithms/Regression/RegressionAlgo.h>
#include <Core/Algorithms/Regression/CompareFields.h> 
#include <Core/Util/Environment.h>
#include <Core/OS/Dir.h>

#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32
#include <unistd.h>
#include <dirent.h>
#endif

namespace SCIRunAlgo {

using namespace SCIRun;

RegressionAlgo::RegressionAlgo(ProgressReporter* pr) :
  AlgoLibrary(pr)
{
}

bool RegressionAlgo::CompareFields(FieldHandle& field1,FieldHandle& field2)
{
  CompareFieldsAlgo algo;
  return (algo.CompareFields(pr_,field1,field2));
}

bool RegressionAlgo::CompareMatrices(MatrixHandle& matrix1, MatrixHandle& matrix2)
{
  // if both have the same pointer or both are zero then they are equal
  if (matrix1.get_rep() == matrix2.get_rep()) return (true);
  
  bool empty1 = false;
  bool empty2 = false;
  
  if (matrix1.get_rep() == 0) 
  {
    empty1 = true;
  }
  else
  {
    if (matrix1->ncols()*matrix1->nrows() == 0) empty1 = true;
  }

  if (matrix2.get_rep() == 0) 
  {
    empty2 = true;
  }
  else
  {
    if (matrix2->ncols()*matrix2->nrows() == 0) empty1 = true;
  }
  
  if (empty1 ==  true && empty2 == true) return (true);
  
  if (empty1 == true) 
  {
    remark("CompareMatrices: Matrix1 is empty and Matrix2 is not");
    return (false);
  }
  if (empty2 == true)
  {
    remark("CompareMatrices: Matrix2 is empty and Matrix1 is not");
    return (false);
  }

  if (matrix1->nrows() != matrix2->nrows())
  {
    remark("CompareMatrices: The number of rows in matrix1 is not equal to the number of rows in matrix2");
    return (false);
  }

  if (matrix1->ncols() != matrix2->ncols())
  {
    remark("CompareMatrices: The number of rows in matrix1 is not equal to the number of rows in matrix2");
    return (false);
  }

  if (matrix1->is_sparse() && matrix2->is_sparse())
  {
    int *rr1, *rr2;
    int *cc1, *cc2;
    double *vv1, *vv2;
    
    int nnz1,nnz2;
    int nrows1,nrows2;
        
    rr1 = matrix1->sparse()->rows;
    rr2 = matrix2->sparse()->rows;
    cc1 = matrix1->sparse()->columns;
    cc2 = matrix2->sparse()->columns;
    vv1 = matrix1->sparse()->a;
    vv2 = matrix2->sparse()->a;
    nrows1 = matrix1->nrows();
    nrows2 = matrix2->nrows();
    
    if ((rr1==0)||(cc1==0)||(vv1==0))
    {
      remark("CompareMatrices: Matrix1 is invalid, one of the pointers in the matrix is zero");
      return (false);
    }

    if ((rr2==0)||(cc2==0)||(vv2==0))
    {
      remark("CompareMatrices: Matrix1 is invalid, one of the pointers in the matrix is zero");
      return (false);
    }
    
    nnz1 = rr1[nrows1];
    nnz2 = rr2[nrows2];
    
    if (nnz1 != nnz2)
    {
      remark("CompareMatrices: Number of non zeros in both matrices is different. One of the matrices may not be properly compressed.");
      return (false);
    }
    
    for (int p=0; p < nrows1; p++) 
    {
      if (rr1[p] != rr2[p])
      {
        remark("CompareMatrices: Number of nonzeros for one or more rows is not equal");
        return (false);        
      }
    }

    for (int p=0; p < nnz1; p++) 
    {
      if (cc1[p] != cc2[p])
      {
        remark("CompareMatrices: Number of nonzeros for one or more columns is not equal");
        return (false);        
      }
    }

    for (int p=0; p < nnz1; p++) 
    {
      if (vv1[p] != vv2[p])
      {
        remark("CompareMatrices: Values in matrices are not equal");
        return (false);        
      }
    }
  }
  else if ((matrix1->is_dense() && matrix2->is_dense())||(matrix1->is_column() && matrix2->is_column())||(matrix1->is_dense_col_maj() && matrix2->is_dense_col_maj()))
  {
    double *data1 = matrix1->get_data_pointer();
    double *data2 = matrix2->get_data_pointer();
    int size = matrix1->get_data_size();
    for (int p=0; p<size; p++) 
      if (data1[p] != data2[p])
      {
        remark("CompareMatrices: Values in matrices are not equal");
        return (false);  
      }
  }
  else
  {
    DenseMatrix *mat1 = matrix1->as_dense();
    DenseMatrix *mat2 = matrix2->as_dense();
  
    double *data1 = mat1->get_data_pointer();
    double *data2 = mat2->get_data_pointer();
    int size = mat1->get_data_size();
    for (int p=0; p<size; p++) 
      if (data1[p] != data2[p])
      {
        remark("CompareMatrices: Values in matrices are not equal");
        return (false);  
      }
  }
  
  return (true);
}

bool RegressionAlgo::CompareNrrds(NrrdDataHandle& nrrd1, NrrdDataHandle& nrrd2)
{
  if (nrrd1.get_rep() == nrrd2.get_rep()) return (true);
  
  if (nrrd1.get_rep() && (nrrd2.get_rep() == 0))
  {
    if (nrrd1->nrrd_ == 0) return (true);
    int dim = nrrd1->nrrd_->dim;
    int size = 1;
    for (int p=0; p<dim; p++) size *= nrrd1->nrrd_->axis[p].size;
    if (size == 0) return (true);
    remark("CompareNrrds: One Nrrd is empty, the other is not");
    return (false);
  }

  if (nrrd2.get_rep() && (nrrd1.get_rep() == 0))
  {
    if (nrrd2->nrrd_ == 0) return (true);
    int dim = nrrd2->nrrd_->dim;
    int size = 1;
    for (int p=0; p<dim; p++) size *= nrrd2->nrrd_->axis[p].size;
    if (size == 0) return (true);
    remark("CompareNrrds: One Nrrd is empty, the other is not");
    return (false);
  }

  if (nrrd1->nrrd_ == nrrd2->nrrd_) return (true);
  
  if (nrrd1->nrrd_ && (nrrd2->nrrd_ == 0))
  {
    int dim = nrrd1->nrrd_->dim;
    int size = 1;
    for (int p=0; p<dim; p++) size *= nrrd1->nrrd_->axis[p].size;
    if (size == 0) return (true);
    remark("CompareNrrds: One Nrrd is empty, the other is not");
    return (false);  
  }
  
  if (nrrd2->nrrd_ && (nrrd1->nrrd_ == 0))
  {
    int dim = nrrd2->nrrd_->dim;
    int size = 1;
    for (int p=0; p<dim; p++) size *= nrrd2->nrrd_->axis[p].size;
    if (size == 0) return (true);
    remark("CompareNrrds: One Nrrd is empty, the other is not");
    return (false);  
  }

  if (nrrd1->nrrd_->dim != nrrd2->nrrd_->dim)
  {
    remark("CompareNrrds: The dimension of the nrrds is not equal");  
    return (false);
  }

  if (nrrd1->nrrd_->type != nrrd2->nrrd_->type)
  {
    remark("CompareNrrds: The type of the nrrds is not equal");  
    return (false);
  }

  if (nrrd1->nrrd_->blockSize != nrrd2->nrrd_->blockSize)
  {
    remark("CompareNrrds: The block size of the nrrds is not equal");  
    return (false);
  }

  int dim1 = nrrd1->nrrd_->dim; 
  int size = 1;
  for (int p=0; p<dim1; p++) 
  {
    if (nrrd1->nrrd_->axis[p].size != nrrd2->nrrd_->axis[p].size)
    {
      remark("CompareNrrds: The axis sizes of the nrrds are not equal");  
      return (false);
    }
    
    if (nrrd1->nrrd_->axis[p].spacing != nrrd2->nrrd_->axis[p].spacing)
    {
      remark("CompareNrrds: The axis spacings of the nrrds are not equal");  
      return (false);
    }
    
    if (nrrd1->nrrd_->axis[p].thickness != nrrd2->nrrd_->axis[p].thickness)
    {
      remark("CompareNrrds: The axis thicknesses of the nrrds are not equal");  
      return (false);
    }

    if (nrrd1->nrrd_->axis[p].min != nrrd2->nrrd_->axis[p].min)
    {
      remark("CompareNrrds: The axis minimums of the nrrds are not equal");  
      return (false);
    }

    if (nrrd1->nrrd_->axis[p].max != nrrd2->nrrd_->axis[p].max)
    {
      remark("CompareNrrds: The axis maximums of the nrrds are not equal");  
      return (false);
    }
    
    size *= nrrd1->nrrd_->axis[p].size;
  }
   
  if (nrrd1->nrrd_->data == 0)
  {
    remark("CompareNrrds: The first nrrd does not have any data in it");  
    return (false);
  }

  if (nrrd2->nrrd_->data == 0)
  {
    remark("CompareNrrds: The second nrrd does not have any data in it");  
    return (false);
  }
   
  if (nrrd1->nrrd_->type == nrrdTypeChar)
  { 
    signed char* data1 = reinterpret_cast<signed char*>(nrrd1->nrrd_->data); 
    signed char* data2 = reinterpret_cast<signed char*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }
  
  if (nrrd1->nrrd_->type == nrrdTypeUChar)
  { 
    unsigned char* data1 = reinterpret_cast<unsigned char*>(nrrd1->nrrd_->data); 
    unsigned char* data2 = reinterpret_cast<unsigned char*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }
  
  if (nrrd1->nrrd_->type == nrrdTypeShort)
  { 
    signed short* data1 = reinterpret_cast<signed short*>(nrrd1->nrrd_->data); 
    signed short* data2 = reinterpret_cast<signed short*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }
  
  if (nrrd1->nrrd_->type == nrrdTypeUShort)
  { 
    unsigned short* data1 = reinterpret_cast<unsigned short*>(nrrd1->nrrd_->data); 
    unsigned short* data2 = reinterpret_cast<unsigned short*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }

  if (nrrd1->nrrd_->type == nrrdTypeInt)
  { 
    signed int* data1 = reinterpret_cast<signed int*>(nrrd1->nrrd_->data); 
    signed int* data2 = reinterpret_cast<signed int*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }
  
  if (nrrd1->nrrd_->type == nrrdTypeUInt)
  { 
    unsigned int* data1 = reinterpret_cast<unsigned int*>(nrrd1->nrrd_->data); 
    unsigned int* data2 = reinterpret_cast<unsigned int*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }

#ifdef _WIN32
	typedef signed __int64 int64;
	typedef unsigned __int64 uint64;
#else
	typedef signed long long int64;
	typedef unsigned long long uint64;
#endif


  if (nrrd1->nrrd_->type == nrrdTypeLLong)
  { 
    int64* data1 = reinterpret_cast<int64*>(nrrd1->nrrd_->data); 
    int64* data2 = reinterpret_cast<int64*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }
  
  if (nrrd1->nrrd_->type == nrrdTypeULLong)
  { 
    uint64* data1 = reinterpret_cast<uint64*>(nrrd1->nrrd_->data); 
    uint64* data2 = reinterpret_cast<uint64*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }

  if (nrrd1->nrrd_->type == nrrdTypeFloat)
  { 
    float* data1 = reinterpret_cast<float*>(nrrd1->nrrd_->data); 
    float* data2 = reinterpret_cast<float*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }
  
  if (nrrd1->nrrd_->type == nrrdTypeDouble)
  { 
    double* data1 = reinterpret_cast<double*>(nrrd1->nrrd_->data); 
    double* data2 = reinterpret_cast<double*>(nrrd2->nrrd_->data); 
    for (int p = 0; p < size; p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }

  if (nrrd1->nrrd_->type == nrrdTypeBlock)
  { 
    char* data1 = reinterpret_cast<char*>(nrrd1->nrrd_->data); 
    char* data2 = reinterpret_cast<char*>(nrrd2->nrrd_->data); 
    for (size_t p = 0; p < size*(nrrd1->nrrd_->blockSize); p++)
    {
      if (data1[p] == data2[p])
      {
        remark("CompareNrrds: The data in the nrrds is not equal");  
        return (false);    
      }
    } 
  }  
  return (true);
}


bool RegressionAlgo::CompareStrings(StringHandle& string1, StringHandle& string2)
{
  if (string1.get_rep() == string2.get_rep())
  {
    return (true);
  }
  
  if (string2.get_rep() == 0)
  {
    if (string1->get() == "") return (true);
    remark("CompareStrings: Strings are not equal");
    return (false);
  }

  if (string1.get_rep() == 0)
  {
    if (string2->get() == "") return (true);
    remark("CompareStrings: Strings are not equal");
    return (false);
  }

  std::string str1 = string1->get();
  std::string str2 = string2->get();

  if (str1 == str2) return (true);

  remark("CompareStrings: Strings are not equal");
  return (false);
}


bool RegressionAlgo::CompareBundles(BundleHandle& bundle1, BundleHandle& bundle2)
{
  if (bundle1.get_rep() == bundle2.get_rep()) return (true);
  if (bundle1.get_rep() == 0)
  {
    remark("CompareBundles: Bundle 1 is empty");
    return (false);
  }
  if (bundle2.get_rep() == 0)
  {
    remark("CompareBundles: Bundle 2 is empty");
    return (false);
  }
  
  int numfields =  bundle1->numFields();
  for (int p = 0; p < numfields; p++)
  {
    std::string name = bundle1->getFieldName(p);
    FieldHandle field1, field2;
    field1 = bundle1->getField(name);
    field2 = bundle2->getField(name);
    
    if (!(CompareFields(field1,field2)))
    {
      remark("CompareBundles: field '"+name+"' is not equal");
      return (false);
    }
  }

  int nummatrices =  bundle1->numMatrices();
  for (int p = 0; p < nummatrices; p++)
  {
    std::string name = bundle1->getMatrixName(p);
    MatrixHandle matrix1, matrix2;
    matrix1 = bundle1->getMatrix(name);
    matrix2 = bundle2->getMatrix(name);
    
    if (!(CompareMatrices(matrix1,matrix2)))
    {
      remark("CompareBundles: matrix '"+name+"' is not equal");
      return (false);
    }
  }

  int numnrrds =  bundle1->numNrrds();
  for (int p = 0; p < numnrrds; p++)
  {
    std::string name = bundle1->getNrrdName(p);
    NrrdDataHandle nrrd1, nrrd2;
    nrrd1 = bundle1->getNrrd(name);
    nrrd2 = bundle2->getNrrd(name);
    
    if (!(CompareNrrds(nrrd1,nrrd2)))
    {
      remark("CompareBundles: nrrd '"+name+"' is not equal");
      return (false);
    }
  }

  int numstrings =  bundle1->numStrings();
  for (int p = 0; p < numstrings; p++)
  {
    std::string name = bundle1->getStringName(p);
    StringHandle string1, string2;
    string1 = bundle1->getString(name);
    string2 = bundle2->getString(name);
    
    if (!(CompareStrings(string1,string2)))
    {
      remark("CompareBundles: string '"+name+"' is not equal");
      return (false);
    }
  }

  return (true);
}


bool RegressionAlgo::FindTestFields(std::vector<FieldHandle> fields, std::vector<FieldHandle>& testfields, std::string fieldtypes)
{

  testfields.clear();
  
  bool meshtype = false;
  bool regular = false;
  bool structured = false;
  bool unstructured = false;
  
  bool fieldbasis = false;
  bool nodata = false;
  bool constantdata = false;
  bool lineardata = false;
  bool nonlineardata = false;
  
  bool meshbasis = false;
  bool linearmesh = false;
  bool nonlinearmesh = false;
  
  bool meshdimension = false;
  bool point = false;
  bool line = false;
  bool surface = false;
  bool volume = false;
  
  bool datatype  = false;
  bool ischar    = false;
  bool isshort   = false;
  bool isinteger = false;
  bool isfloat   = false;
  bool isdouble  = false;
  bool isvector  = false;
  bool istensor  = false;

  bool linearonly = false;
  bool isomorphic = false;

  bool elementtype = false;
  bool pointcloud = false;
  bool scanline   = false;
  bool image      = false;
  bool latvol     = false;
  bool curve      = false;
  bool structcurve = false;
  bool trisurf    = false;
  bool quadsurf   = false;
  bool structquadsurf = false;
  bool tetvol     = false;
  bool prismvol   = false;
  bool hexvol     = false;
  bool structhexvol = false;

  fieldtypes += '|';
  while (1)
  {
    size_t loc = fieldtypes.find('|');
    if (loc > fieldtypes.size()) break;
    std::string fieldtype = fieldtypes.substr(0,loc);
    fieldtypes = fieldtypes.substr(loc+1);

    if (fieldtype == "regular") { meshtype = true; regular = true; }
    if (fieldtype == "structured") { meshtype = true; structured = true; }
    if (fieldtype == "unstructured") { meshtype = true; unstructured = true; }

    if (fieldtype == "nodata") { fieldbasis = true; nodata = true; }
    if (fieldtype == "constantdata") { fieldbasis = true; constantdata = true; }
    if (fieldtype == "lineardata") { fieldbasis = true; lineardata = true; }
    if (fieldtype == "nonlineardata") { fieldbasis = true; nonlineardata = true; }
    
    if (fieldtype == "linearmesh") { meshbasis = true; linearmesh = true; }
    if (fieldtype == "nonlinearmesh") { meshbasis = true; nonlinearmesh = true; }

    if (fieldtype == "point") { meshdimension = true; point = true; }
    if (fieldtype == "line") { meshdimension = true;  line = true; }
    if (fieldtype == "surface") { meshdimension = true; surface = true; }
    if (fieldtype == "volume") { meshdimension = true; volume = true; }
        
    if (fieldtype == "char") { datatype = true; ischar = true; }
    if (fieldtype == "short") { datatype = true; isshort = true; }
    if (fieldtype == "int") { datatype = true; isinteger = true; }
    if (fieldtype == "double") { datatype = true; isdouble = true; }
    if (fieldtype == "flaot") { datatype = true; isfloat = true; }
    if (fieldtype == "vector") { datatype = true; isvector = true; }
    if (fieldtype == "tensor") { datatype = true; istensor = true; }

    if (fieldtype == "linear") linearonly = true;
    if (fieldtype == "isomorphic") isomorphic = true;
    
    if (fieldtype == "pointcloud") { elementtype = true; pointcloud = true; }
    if (fieldtype == "scanline") { elementtype = true; scanline = true; }
    if (fieldtype == "image") { elementtype = true; image = true; }
    if (fieldtype == "latvol") { elementtype = true; latvol = true; }
    if (fieldtype == "curve") { elementtype = true; curve = true; }
    if (fieldtype == "structcurve") { elementtype = true; structcurve = true; }
    if (fieldtype == "trisurf") { elementtype = true; trisurf = true; }
    if (fieldtype == "quadsurf") { elementtype = true; quadsurf = true; }
    if (fieldtype == "structquadsurf") { elementtype = true; structquadsurf = true; }
    if (fieldtype == "tetvol") { elementtype = true; tetvol = true; }
    if (fieldtype == "prismvol") { elementtype = true; prismvol = true; }
    if (fieldtype == "hexvol") { elementtype = true; hexvol = true; }
    if (fieldtype == "structhexvol") { elementtype = true; structhexvol = true; }

  }
  
  
  for (unsigned int p = 0; p < fields.size(); p++)
  {
    FieldInformation fi(fields[p]);

    if (elementtype)
    {
      if (fi.is_pointcloud() && !pointcloud) continue;
      if (fi.is_scanline() && !scanline) continue;
      if (fi.is_image() && !image) continue;
      if (fi.is_latvol() && !latvol) continue;
      if (fi.is_curve() && !curve) continue;
      if (fi.is_structcurve() && !structcurve) continue;
      if (fi.is_trisurf() && !trisurf) continue;
      if (fi.is_quadsurf() && !quadsurf) continue;
      if (fi.is_structquadsurf() && !structquadsurf) continue;
      if (fi.is_tetvol() && !tetvol) continue;
      if (fi.is_prismvol() && !prismvol) continue;
      if (fi.is_hexvol() && !hexvol) continue;
      if (fi.is_structhexvol() && !structhexvol) continue;
    }
    
    if (linearonly && fi.is_nonlinear()) continue;
    if (meshtype)
    {
      if (fi.is_regularmesh() && !regular) continue;
      if (fi.is_structuredmesh() && !structured) continue;
      if (fi.is_unstructuredmesh() && !unstructured) continue;
    }
    
    if (fieldbasis)
    {
      if (fi.is_nodata() && !nodata) continue;
      if (fi.is_constantdata() && !constantdata) continue;
      if (fi.is_lineardata() && !lineardata) continue;
      if (fi.is_nonlineardata() && !nonlineardata) continue;
    }
  
    if (meshbasis)
    {
      if (fi.is_linearmesh() && !linearmesh) continue;
      if (fi.is_nonlinearmesh() && !nonlinearmesh) continue;
    }
 
    if (meshdimension)
    {
      if (fi.is_point() && !point) continue;
      if (fi.is_line() && !line) continue;
      if (fi.is_surface() && !surface) continue;
      if (fi.is_volume() && !volume) continue;
    }
  
    if (datatype)
    {
      if (fi.is_char() && !ischar) continue;
      if (fi.is_short() && !isshort) continue;
      if (fi.is_integer() && !isinteger) continue;
      if (fi.is_float() && !isfloat) continue;
      if (fi.is_double() && !isdouble) continue;
      if (fi.is_vector() && !isvector) continue;
      if (fi.is_tensor() && !istensor) continue;
    }
  
    if (isomorphic) if (!fi.is_isomorphic()) continue;
  
    testfields.push_back(fields[p]);
  }

  return (true);
}


bool RegressionAlgo::FindTestField(std::vector<FieldHandle> fields, FieldHandle& testfield, std::string name)
{
  for (unsigned int p =0; p< fields.size(); p++)
  {
    if (fields[p]->get_name() == name)
    {
      testfield = fields[p];
      return (true);
    }
  }
  return (false);
}

bool RegressionAlgo::LoadTestFields(std::vector<FieldHandle>& fields)
{

  const char *regdir = sci_getenv("SCIRUN_REGRESSION_DIR");

  std::string flddir = std::string(regdir) + "/testfields";
  std::vector<std::string> files;
  std::vector<std::string> fieldnames;

  DIR* dir = opendir(flddir.c_str());
  if (dir)
  {
   dirent* file = readdir(dir);
   while (file)
   {
     std::string filename(file->d_name);
     if (filename.size() > 4)
     {
       if (filename.substr(filename.size()-4,4) == std::string(".fld"))
       {
        files.push_back(flddir + "/" + filename);
        fieldnames.push_back(filename.substr(0,filename.size()-4));
       }
     }
     file = readdir(dir);
   }
   closedir(dir);
  }
    
  SCIRunAlgo::DataIOAlgo dalgo(pr_);
              
  for (size_t p = 0; p < files.size(); p++)
  {
    FieldHandle field;
    if(dalgo.ReadField(files[p],field))
    {
      field->set_name(fieldnames[p]);
      fields.push_back(field);
    }
    else
    {
      pr_->error("LoadTestFields: Could not read file '"+files[p] +"'");
      return (false);
    }
  }
  
  return (true);
}


} // end namespace SCIRunAlgo

