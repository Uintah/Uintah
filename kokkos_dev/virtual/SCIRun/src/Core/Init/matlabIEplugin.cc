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

#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
#include <Core/ImportExport/Nrrd/NrrdIEPlugin.h>
#include <Core/Matlab/matlabfile.h>
#include <Core/Matlab/matlabarray.h>
#include <Core/Matlab/matlabconverter.h>


namespace SCIRun {

using namespace MatlabIO;

// SparseRowMatrix
MatrixHandle
MatlabMatrix_reader(ProgressReporter *pr, const char *filename)
{
  matlabfile mf;
  matlabconverter mc(pr);
  matlabarray ma;
  int numarrays;
  std::string dummytext;
  MatrixHandle mh;
  
  mh = 0;
  
  try
  {
      mf.open(std::string(filename),"r");
      numarrays = mf.getnummatlabarrays();
      for (int p=0;p<numarrays;p++)
      {
        ma = mf.getmatlabarrayinfo(p);
        if (mc.sciMatrixCompatible(ma,dummytext)) 
        { 
          ma = mf.getmatlabarray(p);
          mc.mlArrayTOsciMatrix(ma,mh); break; 
        }
      }
      mf.close();
  }
  catch (...)
  {
    mh = 0;
  }

  return(mh);
}

bool
MatlabMatrix_writer(ProgressReporter *pr,
			   MatrixHandle mh, const char *filename)
{
  matlabfile mf;
  matlabconverter mc(pr);
  matlabarray ma;
  std::string name;
 
  try
  {
      mc.converttonumericmatrix();
      mc.sciMatrixTOmlArray(mh,ma);
      mh->get_property("name",name);
      if ((name=="")||(!mc.isvalidmatrixname(name))) name = "scirunmatrix";
      mf.open(std::string(filename),"w");
      mf.putmatlabarray(ma,name);
      mf.close();
  }
  catch (...)
  {
    return(false);
  }
  return(true);
}
  


NrrdDataHandle
MatlabNrrd_reader(ProgressReporter *pr, const char *filename)
{
  matlabfile mf;
  matlabconverter mc(pr);
  matlabarray ma;
  int numarrays;
  std::string dummytext;
  NrrdDataHandle mh;
  
  mh = 0;
  
  try
  {
      mf.open(std::string(filename),"r");
      numarrays = mf.getnummatlabarrays();
      for (int p=0;p<numarrays;p++)
      {
        ma = mf.getmatlabarrayinfo(p);
        if (mc.sciNrrdDataCompatible(ma,dummytext)) 
        { 
          ma = mf.getmatlabarray(p);
          mc.mlArrayTOsciNrrdData(ma,mh); break; 
        }
      }
      mf.close();
  }
  catch (...)
  {
    mh = 0;
  }

  return(mh);
}

bool
MatlabNrrd_writer(ProgressReporter *pr,
			   NrrdDataHandle mh, const char *filename)
{
  matlabfile mf;
  matlabconverter mc(pr);
  matlabarray ma;
  std::string name;
 
  try
  {
      mc.converttonumericmatrix();
      mc.sciNrrdDataTOmlArray(mh,ma);
      mh->get_property("name",name);
      if ((name=="")||(!mc.isvalidmatrixname(name))) name = "scirunnrrd";
      mf.open(std::string(filename),"w");
      mf.putmatlabarray(ma,name);
      mf.close();
  }
  catch (...)
  {
    return(false);
  }
  return(true);
}

         
FieldHandle
MatlabField_reader(ProgressReporter *pr, const char *filename)
{
  matlabfile mf;
  matlabconverter mc(pr);
  matlabarray ma;
  int numarrays;
  std::string dummytext;
  FieldHandle mh;
  
  mh = 0;
  
  try
  {
      mf.open(std::string(filename),"r");
      numarrays = mf.getnummatlabarrays();
      for (int p=0;p<numarrays;p++)
      {
        ma = mf.getmatlabarrayinfo(p);
        if (mc.sciFieldCompatible(ma,dummytext)) 
        {
          ma = mf.getmatlabarray(p);
          mc.mlArrayTOsciField(ma,mh); break; 
        }
      }
      mf.close();
  }
  catch (...)
  {
    mh = 0;
  }

  return(mh);
}

bool
MatlabField_writer(ProgressReporter *pr,
			   FieldHandle mh, const char *filename)
{
  matlabfile mf;
  matlabconverter mc(pr);
  matlabarray ma;
  std::string name;
 
  try
  {
      mc.converttostructmatrix();
      mc.sciFieldTOmlArray(mh,ma);
      mh->get_property("name",name);
      if ((name=="")||(!mc.isvalidmatrixname(name))) name = "scirunfield";
      mf.open(std::string(filename),"w");
      mf.putmatlabarray(ma,name);
      mf.close();
  }
  catch (...)
  {
    // there is no way to signal an error to a module upstream
    return(false);
  }
  return(true);
}

static MatrixIEPlugin MatlabMatrix_plugin("Matlab Matrix",".mat", "*.mat", MatlabMatrix_reader, MatlabMatrix_writer);
static FieldIEPlugin MatlabField_plugin("Matlab Field",".mat", "*.mat",MatlabField_reader,MatlabField_writer);   
static NrrdIEPlugin MatlabNrrd_plugin("Matlab Matrix",".mat", "*.mat",MatlabNrrd_reader,MatlabNrrd_writer);     
  
} // end namespace SCIRun

