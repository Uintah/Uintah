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
 * FILE: StreamMatrixAlgo.h
 * AUTH: Jeroen G Stinstra
 * DATE: 23 MAR 2005
 */
 
 
#ifndef PACKAGES_MODELCREATION_CORE_DATASTREAMING_STREAMMATRIXALGO_H
#define PACKAGES_MODELCREATION_CORE_DATASTREAMING_STREAMMATRIXALGO_H 1

#include <Core/Thread/Time.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Exceptions/Exception.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace ModelCreation {

class StreamMatrixAlgo {

  public:
  
    // Constructor of the datastreaming class
    StreamMatrixAlgo(SCIRun::ProgressReporter* pr);
	  StreamMatrixAlgo(SCIRun::ProgressReporter* pr, std::string filename);
    
    // Destructor of the datastreaming class
    virtual ~StreamMatrixAlgo();
    
    // Open and close file for datastreaming
    bool open(std::string filename);
    bool close();
	
    // Get the essential information from the data file
	  int         get_numrows();
	  int         get_numcols();
    double      get_rowspacing();
    double      get_colspacing();
    std::string get_rowkind(); 
    std::string get_colkind(); 
    std::string get_rowunit();
    std::string get_colunit();
	
  	// Get the content and unit of the data
    std::string get_content();    
    std::string get_sampleunit(); 

    // Get the columns from the matrix	
    bool getcolmatrix(SCIRun::MatrixHandle& colmatrix, SCIRun::MatrixHandle indices);
    bool getcolmatrix_weights(SCIRun::MatrixHandle& colmatrix, SCIRun::MatrixHandle weights);

    bool getrowmatrix(SCIRun::MatrixHandle& rowmatrix, SCIRun::MatrixHandle indices);
    bool getrowmatrix_weights(SCIRun::MatrixHandle& rowmatrix, SCIRun::MatrixHandle weights);
	
    
  private:  
	  // Remove white spaces from data when reading the header file
    std::string remspaces(std::string str); 
	
	  // Get the nrrdtype and the element size
    void gettype(std::string type,int& nrrdtype, int& elsize); 
	
	  // Nrrd header is text, make it case insensitive
    int  cmp_nocase(const std::string& s1, const std::string& s2);  
    
	  // Swap bytes if needed, the software has to work on different processor types.
	  void doswapbytes(void *vbuffer,long elsize,long size); 

    SCIRun::ProgressReporter* pr_;

    std::string               datafilename_;    // if there is one file use this one
    std::vector<std::string>  datafilenames_;   // if data is spread over multiple files use this one
    std::vector<std::string>  units_;           // Unit per axis
    std::vector<std::string>  kinds_;           // Kind per axis
    std::vector<double>       spacings_;        // Spacing per axis
    std::vector<int>          sizes_;           // Size per axis
    std::vector<int>          coloffset_;       // Internal use only

    std::string               content_;         // Content of the data file
    std::string               encoding_;        // Needs to be raw, as we cannot deal with compressed files yet
    std::string               endian_;          // Endian information
    std::string               type_;
    std::string               unit_;
    int                       lineskip_;
    int                       byteskip_;
    int                       elemsize_;
    int                       ntype_;
    int                       dimension_;
    
    int                       start_;
    int                       end_;
    int                       step_;
    int                       subdim_;
    
    bool                      useformatting_;
    bool                      swapbytes_;
  
  };
  
}

#endif
