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
 * FILE: cardiowaveconverter.h
 * AUTH: Jeroen G Stinstra
 * DATE: 1 FEB 2005
 */
 
#ifndef JGS_MATLABIO_CARDIOWAVECONVERTER_H
#define JGS_MATLABIO_CARDIOWAVECONVERTER_H 1

#include <stdio.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream> 
#include <sgi_stl_warnings_on.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Dataflow/Network/Module.h>


#include <Core/ImportExport/Matrix/MatrixIEPlugin.h>
#include <Core/ImportExport/Field/FieldIEPlugin.h>

#ifdef _WIN32
	typedef signed __int64 int64;
	typedef unsigned __int64 uint64;
#else
	typedef signed long long int64;
	typedef unsigned long long uint64;
#endif

using namespace SCIRun;

namespace CardioWave {

class CardioWaveConverter {
  public:
  
    // constructor
    CardioWaveConverter();
  
    // Functions for loading CardioWave files
    bool cwFileTOsciMatrix(std::string filename,MatrixHandle& mh,ProgressReporter *pr);
    bool sciMatrixTOcwFile(MatrixHandle mh,std::string filename,ProgressReporter *pr,std::string filetype = "");
  
  private:

    // Byte swapping stuff
    bool byteswapmachine();
    void swapbytes(void *buffer,int elsize,int size);
    
    void posterror(ProgressReporter *pr,string msg);
    void postwarning(ProgressReporter *pr,string msg);
    
};


}

#endif
