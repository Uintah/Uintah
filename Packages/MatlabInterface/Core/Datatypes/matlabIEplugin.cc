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

#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>
// #include <Core/ImportExprt/Nrrd/NrrdIEPlugin.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabfile.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabarray.h>
#include <Packages/MatlabInterface/Core/Datatypes/matlabconverter.h>

using namespace std;
using namespace SCIRun;
using namespace MatlabIO;

// SparseRowMatrix
MatrixHandle
MatlabMatrix_reader(ProgressReporter *pr, const char *filename)
{
  matlabfile mf;
  matlabconverter mc;
  matlabarray ma;
  long numarrays;
  string dummytext;
  MatrixHandle mh;
  
  mh = 0;
  
  try
  {
      mf.open(std::string(filename),"r");
      numarrays = mf.getnummatlabarrays();
      for (long p=0;p<numarrays;p++)
      {
        ma = mf.getmatlabarrayinfo(p);
        if (mc.sciMatrixCompatible(ma,dummytext,pr)) 
        { 
          ma = mf.getmatlabarray(p);
          mc.mlArrayTOsciMatrix(ma,mh,pr); break; 
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
  matlabconverter mc;
  matlabarray ma;
  string name;
 
  try
  {
      mc.converttonumericmatrix();
      mc.sciMatrixTOmlArray(mh,ma,pr);
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
  matlabconverter mc;
  matlabarray ma;
  long numarrays;
  string dummytext;
  NrrdDataHandle mh;
  
  mh = 0;
  
  try
  {
      mf.open(std::string(filename),"r");
      numarrays = mf.getnummatlabarrays();
      for (long p=0;p<numarrays;p++)
      {
        ma = mf.getmatlabarrayinfo(p);
        if (mc.sciNrrdDataCompatible(ma,dummytext,pr)) 
        { 
          ma = mf.getmatlabarray(p);
          mc.mlArrayTOsciNrrdData(ma,mh,pr); break; 
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
  matlabconverter mc;
  matlabarray ma;
  string name;
 
  try
  {
      mc.converttonumericmatrix();
      mc.sciNrrdDataTOmlArray(mh,ma,pr);
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
  matlabconverter mc;
  matlabarray ma;
  long numarrays;
  string dummytext;
  FieldHandle mh;
  
  mh = 0;
  
  try
  {
      mf.open(std::string(filename),"r");
      numarrays = mf.getnummatlabarrays();
      for (long p=0;p<numarrays;p++)
      {
        ma = mf.getmatlabarrayinfo(p);
        if (mc.sciFieldCompatible(ma,dummytext,pr)) 
        {
          ma = mf.getmatlabarray(p);
          mc.mlArrayTOsciField(ma,mh,pr); break; 
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
  matlabconverter mc;
  matlabarray ma;
  string name;
 
  try
  {
      mc.converttostructmatrix();
      mc.sciFieldTOmlArray(mh,ma,pr);
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

     
         

