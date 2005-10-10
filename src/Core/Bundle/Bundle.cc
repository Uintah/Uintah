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


#include <Core/Bundle/Bundle.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>

using namespace SCIRun;
using namespace std;

static Persistent* make_Bundle() {
  return scinew Bundle;
}

PersistentTypeID Bundle::type_id("Bundle", "PropertyManager", make_Bundle);

Bundle::Bundle()
{
}

Bundle::~Bundle() 
{
}

Bundle* Bundle::clone() 
{
  return(scinew Bundle(*this));
}


int Bundle::findName(deque<string> &deq,string name)
{
  for (unsigned int p = 0; p<deq.size(); p++)
  {
    if (cmp_nocase(name,deq[p]) == 0) return p;
  }
  return(-1);
}

int Bundle::cmp_nocase(const string &s1,const string &s2)
{
  string::const_iterator p1 = s1.begin();
  string::const_iterator p2 = s2.begin();
	
  while (p1!=s1.end() && p2 != s2.end())
  {
    if (toupper(*p1) != toupper(*p2)) 
        return ((toupper(*p1) < toupper(*p2)) ? -1 : 1);
    p1++; p2++;
  }
	
  return((s2.size()==s1.size()) ? 0 : (s1.size() < s2.size()) ? -1 :  1);
}


void Bundle::merge(LockingHandle<Bundle> C)
{
  for (size_t p = 0; p < C->bundle_.size(); p++)
  {
    string name = C->bundleName_[p];
    LockingHandle<PropertyManager> handle = C->bundle_[p];
      
    int index;
    index = findName(bundleName_,name);
    if (index == -1)
    {
      bundle_.push_back(handle);
      bundleName_.push_back(name);
    }
    else
    {
      bundle_[index] = handle.get_rep();
      bundleName_[index] = name;
    }
  }
}

#define BUNDLE_VERSION 2

//////////
// PIO for NrrdData objects
void
Bundle::io(Piostream& stream)
{
  stream.begin_class("Bundle", BUNDLE_VERSION);
  // Do the base class first...
 
  PropertyManager::io(stream);
  
  if (stream.reading()) 
  {
    int size;

    stream.begin_cheap_delim();
      
    stream.io(size);

    bundle_.resize(size);
    bundleName_.resize(size);
  
    string type;
    for (int p = 0; p < size; p++)
    {
      stream.begin_cheap_delim();
      stream.io(bundleName_[p]);
      stream.io(type);
      stream.end_cheap_delim();		
      stream.begin_cheap_delim();
      if (type=="field")
      {
        LockingHandle<Field> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="matrix")
      {
        LockingHandle<Matrix> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="string")
      {
        LockingHandle<String> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="nrrd")
      {
        LockingHandle<NrrdData> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="colormap")
      {
        LockingHandle<ColorMap> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="colormap2")
      {
        LockingHandle<ColorMap2> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="path")
      {
        LockingHandle<Path> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      if (type=="bundle")
      {
        LockingHandle<Bundle> handle;
        Pio(stream,handle);
        bundle_[p] = dynamic_cast<PropertyManager *>(handle.get_rep());
      }
      stream.end_cheap_delim();
    }
    stream.end_cheap_delim();		
  } 
  else 
  { 
    size_t size, tsize;
    stream.begin_cheap_delim();
    tsize = 0;
    size = bundleName_.size();
    for (size_t p = 0; p < size; p ++)
    {
      if (bundle_[p].get_rep()) tsize++;
    }

    stream.io(tsize);
    for (size_t p = 0; p < size; p++)
    {
      if (bundle_[p].get_rep())
      {
        stream.begin_cheap_delim();		
        stream.io(bundleName_[p]);

        string type;
        LockingHandle<Field> fieldhandle = 
                                    dynamic_cast<Field*>(bundle_[p].get_rep());
        if (fieldhandle.get_rep()) 
        { 
          type = "field";
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,fieldhandle); 
          stream.end_cheap_delim(); 
          continue; 
        }
      
        LockingHandle<Matrix> matrixhandle = 
                                   dynamic_cast<Matrix*>(bundle_[p].get_rep());
        if (matrixhandle.get_rep()) 
        { 
          type = "matrix";
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,matrixhandle); 
          stream.end_cheap_delim(); 
          continue; 
        }
        
        LockingHandle<String> stringhandle = 
                                   dynamic_cast<String*>(bundle_[p].get_rep());
        if (stringhandle.get_rep()) 
        { 
          type = "string";
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,stringhandle); 
          stream.end_cheap_delim(); 
          continue; 
        }
    
                  
        LockingHandle<NrrdData> nrrdhandle = 
                                 dynamic_cast<NrrdData*>(bundle_[p].get_rep());
        if (nrrdhandle.get_rep())
        { 
          type = "nrrd";
          bool embed_old = nrrdhandle->get_embed_object();
          nrrdhandle->set_embed_object(true);
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,nrrdhandle); 
          stream.end_cheap_delim(); 
          nrrdhandle->set_embed_object(embed_old);
          continue; 
        }
    
      
        LockingHandle<ColorMap> colormaphandle = 
                                 dynamic_cast<ColorMap*>(bundle_[p].get_rep());
        if (colormaphandle.get_rep()) 
        { 
          type = "colormap"; 
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,colormaphandle); 
          stream.end_cheap_delim(); 
          continue; 
        }

        LockingHandle<ColorMap2> colormap2handle = 
                                dynamic_cast<ColorMap2*>(bundle_[p].get_rep());
        if (colormap2handle.get_rep()) 
        { 
          type = "colormap2"; 
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,colormap2handle); 
          stream.end_cheap_delim(); 
          continue; 
        }

        LockingHandle<Path> pathhandle = 
                                    dynamic_cast<Path*>(bundle_[p].get_rep());
        if (pathhandle.get_rep()) 
        { 
          type = "path"; 
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,pathhandle); 
          stream.end_cheap_delim(); 
          continue; 
        }

        LockingHandle<Bundle> bundlehandle = 
                                  dynamic_cast<Bundle*>(bundle_[p].get_rep());
        if (bundlehandle.get_rep()) 
        { 
          type = "bundle";
          stream.io(type);
          stream.end_cheap_delim();		
          stream.begin_cheap_delim();            
          Pio(stream,bundlehandle); 
          stream.end_cheap_delim(); 
          continue; 
        }
      }
    }
    stream.end_cheap_delim();		
  }
} 


bool Bundle::NrrdToMatrixConvertible(NrrdDataHandle nrrdH)
{
  if (nrrdH.get_rep() == 0) return(false);
  if (nrrdH->nrrd == 0) return(false);
  switch (nrrdH->nrrd->type)
  {
  case nrrdTypeChar: case nrrdTypeUChar:
  case nrrdTypeShort: case nrrdTypeUShort:
  case nrrdTypeInt: case nrrdTypeUInt:
  case nrrdTypeFloat: case nrrdTypeDouble:
    break;
  default:
    return(false);
  }
  if (nrrdH->nrrd->dim < 3) return(true);
  return(false);
}


bool
Bundle::NrrdToMatrix(NrrdDataHandle nrrdH,MatrixHandle& matH)
{
  if (nrrdH.get_rep() == 0) return false;
  if (nrrdH->nrrd == 0) return false;
  switch(nrrdH->nrrd->type)
    {
    case nrrdTypeChar:
      return(NrrdToMatrixHelper<char>(nrrdH,matH));
    case nrrdTypeUChar:
      return(NrrdToMatrixHelper<unsigned char>(nrrdH,matH));
    case nrrdTypeShort:
      return(NrrdToMatrixHelper<short>(nrrdH,matH));
    case nrrdTypeUShort:
      return(NrrdToMatrixHelper<unsigned short>(nrrdH,matH));
    case nrrdTypeInt:
      return(NrrdToMatrixHelper<int>(nrrdH,matH));
    case nrrdTypeUInt:
      return(NrrdToMatrixHelper<unsigned int>(nrrdH,matH));
    case nrrdTypeFloat:
      return(NrrdToMatrixHelper<float>(nrrdH,matH));
    case nrrdTypeDouble:
      return(NrrdToMatrixHelper<double>(nrrdH,matH));
    default:
      return(false);
    }
  return(false);
}

bool Bundle::MatrixToNrrdConvertible(MatrixHandle matH)
{
  if (matH.get_rep() == 0) return(false);
  if (matH->is_dense()) return(true);
  if (matH->is_column()) return(true);
  return(false);
}

bool Bundle::MatrixToNrrd(MatrixHandle matH,NrrdDataHandle &nrrdH)
{
  if (matH.get_rep() == 0) return false;
  if (matH->is_dense()) 
  {
    Matrix* matrixptr = matH.get_rep();
    DenseMatrix* matrix = dynamic_cast<DenseMatrix*>(matrixptr);

    int rows = matrix->nrows();
    int cols = matrix->ncols();

    if (transposeNrrd_)
    {
      nrrdH = scinew NrrdData();
      nrrdAlloc(nrrdH->nrrd, nrrdTypeDouble, 2, cols, rows);
      nrrdAxisInfoSet(nrrdH->nrrd, nrrdAxisInfoLabel, 
                                          "dense-columns" , "dense-rows");
      nrrdH->nrrd->axis[0].kind = nrrdKindDomain;
      nrrdH->nrrd->axis[1].kind = nrrdKindDomain;

      double *val = (double*)nrrdH->nrrd->data;
      double *data = matrix->get_data_pointer();

      int i,j;
      i = 0;
      j = 0;
      for(int r=0; r<rows; r++) 
        {
          for(int c=0; c<cols; c++) 
            {
              i = c + cols*r;
              val[i] = data[j++];
            }
        }
    }
    else
    {
      nrrdH = scinew NrrdData();
      nrrdAlloc(nrrdH->nrrd, nrrdTypeDouble, 2, rows, cols);
      nrrdAxisInfoSet(nrrdH->nrrd, nrrdAxisInfoLabel, 
                                          "dense-rows" , "dense-columns");
      nrrdH->nrrd->axis[0].kind = nrrdKindDomain;
      nrrdH->nrrd->axis[1].kind = nrrdKindDomain;

      double *val = (double*)nrrdH->nrrd->data;
      double *data = matrix->get_data_pointer();

      for(int c=0; c<cols; c++) 
        {
          for(int r=0; r<rows; r++) 
            {
              *val++ = *data++;
            }
        }
    
    }
    return(true);
  } 
  else if (matH->is_column()) 
  {
    Matrix* matrixptr = matH.get_rep();
    ColumnMatrix* matrix = dynamic_cast<ColumnMatrix*>(matrixptr);
    int size = matrix->nrows();

    nrrdH = scinew NrrdData();
    nrrdAlloc(nrrdH->nrrd, nrrdTypeDouble, 1, size);
    nrrdAxisInfoSet(nrrdH->nrrd, nrrdAxisInfoLabel, "column-data");
    nrrdH->nrrd->axis[0].kind = nrrdKindDomain;

    double *val = (double*)nrrdH->nrrd->data;
    double *data = matrix->get_data();

    for(int i=0; i<size; i++) 
    {
      *val = *data;
      ++data;
      ++val;
    }
    return(true);
  } 
  else 
  {
    // For the moment we do not convert this one
    // This is the SPARSE matrix one
    return(false);
  }
} // end MatrixToNrrd


LockingHandle<Matrix>
Bundle::getMatrix(string name) 
{ 
  MatrixHandle matrix;
  matrix = get<Matrix>(name);
  if (matrix.get_rep() == 0)
  {
    NrrdDataHandle nrrd;
    nrrd = get<NrrdData>(name);
    if (nrrd.get_rep())
    {
      if (NrrdToMatrixConvertible(nrrd))
        NrrdToMatrix(nrrd,matrix);
    }
  }
  return matrix;
}

bool Bundle::isMatrix(string name)  
{ 
  bool isMat;
  isMat = is<Matrix>(name);
  if (!isMat)
  {
    if (is<NrrdData>(name))
    {
      NrrdDataHandle nrrdH = get<NrrdData>(name);
      if (NrrdToMatrixConvertible(nrrdH)) isMat = true;
    }
  } 
  return(isMat);
}

int Bundle::numMatrices() 
{ 
  int nummat;
  nummat = num<Matrix>();
    
  int numnrrd; 
  numnrrd = num<NrrdData>();
  string name;
  for (int p=0;p < numnrrd; p++)
  {
    name = getName<NrrdData>(p);
    if (isMatrix(name)) nummat++;
  }
  return(nummat);
}

string Bundle::getMatrixName(int index) 
{
  int nummat = num<Matrix>();
  if (index < nummat) return(getName<Matrix>(index));

  int numnrrd; 
  numnrrd = num<NrrdData>();
  for (int p=0;p < numnrrd; p++)
  {
    string name = getName<NrrdData>(p);
    if (isMatrix(name))
    {
      if (index == nummat) return name;
      nummat++;
    }
  }
  return("");
}

string Bundle::getHandleType(int index)
{
  LockingHandle<PropertyManager> handle = gethandle(index);

  if ((dynamic_cast<Field *>(handle.get_rep()))) return("field");
  if ((dynamic_cast<Matrix *>(handle.get_rep()))) return("matrix");
  if ((dynamic_cast<NrrdData *>(handle.get_rep()))) return("nrrd");
  if ((dynamic_cast<ColorMap *>(handle.get_rep()))) return("colormap");
  if ((dynamic_cast<ColorMap2 *>(handle.get_rep()))) return("colormap2");
  if ((dynamic_cast<Path *>(handle.get_rep()))) return("path");
  if ((dynamic_cast<Bundle *>(handle.get_rep()))) return("bundle");
  if ((dynamic_cast<String *>(handle.get_rep()))) return("string");

  return("unknown");
}


LockingHandle<NrrdData> Bundle::getNrrd(string name) 
{ 
  NrrdDataHandle nrrd;
  nrrd = get<NrrdData>(name);
  if (nrrd.get_rep() == 0)
  {
    MatrixHandle matrix;
    matrix = get<Matrix>(name);
    if (matrix.get_rep())
    {
      if (MatrixToNrrdConvertible(matrix))
                      MatrixToNrrd(matrix,nrrd);
    }
  }
  return(nrrd);
}

bool
Bundle::isNrrd(string name)  
{ 
  bool isnrrd;
  isnrrd = is<NrrdData>(name);
  if (!isnrrd)
  {
    if (is<Matrix>(name))
    {
      MatrixHandle matrixH = get<Matrix>(name);
      if (MatrixToNrrdConvertible(matrixH)) isnrrd = true;
    }
  } 
  return(isnrrd);
}

int Bundle::numNrrds() 
{ 
  int numnrrd;
  numnrrd = num<NrrdData>();
    
  int nummat; 
  nummat = num<Matrix>();
  string name;
  for (int p=0;p < nummat; p++)
  {
    name = getName<Matrix>(p);
    if (isNrrd(name)) numnrrd++;
  }
  return(numnrrd);
}

string Bundle::getNrrdName(int index) 
{
  int numnrrd = num<NrrdData>();
  if (index < numnrrd) return(getName<NrrdData>(index));

  int nummat; 
  nummat = num<Matrix>();
  for (int p=0;p < nummat; p++)
  {
    string name = getName<Matrix>(p);
    if (isNrrd(name))
    {
      if (index == numnrrd) return(name);
      numnrrd++;
    }
  }
  return("");
}




