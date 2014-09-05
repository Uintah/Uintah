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
#include <Packages/CardioWave/Core/Datatypes/CardioWaveConverter.h>

using namespace std;
using namespace SCIRun;

namespace CardioWave {

// SparseRowMatrix
MatrixHandle
CWMatrix_reader(ProgressReporter *pr, const char *filename)
{
  CardioWaveConverter cc;
  MatrixHandle mh;
  mh = 0;
  
  try
  {
    if (!(cc.cwFileTOsciMatrix(string(filename),mh,pr)))
    {
      mh = 0;
    }
  }
  catch (...)
  {
    mh = 0;
  }

  return(mh);
}



bool CWMatrix_writer_bvec(ProgressReporter *pr, MatrixHandle mh, const char *filename)
{
  CardioWaveConverter cc;
  
  try
  {
    if (!(cc.sciMatrixTOcwFile(mh,string(filename),pr,"bvec")))
    {
      return(false);
    }
  }
  catch (...)
  {
    return(false);
  }
  return(true);
}

bool CWMatrix_writer_ivec(ProgressReporter *pr, MatrixHandle mh, const char *filename)
{
  CardioWaveConverter cc;
  
  try
  {
    if (!(cc.sciMatrixTOcwFile(mh,string(filename),pr,"ivec")))
    {
      return(false);
    }
  }
  catch (...)
  {
    return(false);
  }
  return(true);
}

bool CWMatrix_writer_vec(ProgressReporter *pr, MatrixHandle mh, const char *filename)
{
  CardioWaveConverter cc;
  
  try
  {
    if (!(cc.sciMatrixTOcwFile(mh,string(filename),pr,"vec")))
    {
      return(false);
    }
  }
  catch (...)
  {
    return(false);
  }
  return(true);
}

bool CWMatrix_writer_spr(ProgressReporter *pr, MatrixHandle mh, const char *filename)
{
  CardioWaveConverter cc;
  
  try
  {
    if (!(cc.sciMatrixTOcwFile(mh,string(filename),pr,"spr")))
    {
      return(false);
    }
  }
  catch (...)
  {
    return(false);
  }
  return(true);
}

}
