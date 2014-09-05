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


// NOTE: This MatlabIO file is used in different projects as well. Please, do not
// make it depend on other scirun code. This way it is easier to maintain matlabIO 
// code among different projects. Thank you.


/*
 * FILE: matlabfile.h
 * AUTH: Jeroen G Stinstra
 * DATE: 16 MAY 2005
 */
 

/*
 * CLASS DESCRIPTION
 * This class is an interface to a matfile and handles the importing and exporting
 * matlabarray objects. The latter represent the full complexity of a matlab array.
 *
 * MEMORY MODEL
 * The class maintains its own copies of the data. Each vector, string and other
 * data unit is copied.
 * Large quantities of data are shipped in and out as matlabarray objects. These
 * objects are handles to complex structures in memory and maintain their own data integrity. 
 * When copying a matfiledata object only pointers are copied, however all information
 * for freeing the object is stored inside and no memory losses should occur.
 *
 * ERROR HANDLING
 * All errors are reported as exceptions described in the matfilebase class.
 * Errors of external c library functions are caught and forwarded as exceptions.
 *
 * COPYING/ASSIGNMENT
 * Do not copy the object, this will lead to errors (NEED TO FIX THIS)
 *
 * RESOURCE ALLOCATION
 * Files are closed by calling close() or by destroying the object
 *
 */
 
 
#ifndef JGS_MATLABIO_MATLABFILE_H
#define JGS_MATLABIO_MATLABFILE_H 1


#include "matfilebase.h"
#include "matfile.h"
#include "matfiledata.h"
#include "matlabarray.h"

#include <vector>
#include <string>
#include <iostream>
 
namespace MatlabIO {

  class matlabfile : public matfile {
  
  private:
    // matrixaddress is a vector of offsets
    // where the different matrices are stored
    // matrixname is a vector of the same length
    // storing the name of these matrices
        
    // NOTE: These fields are only available for
    // read access
    std::vector<long> matrixaddress_;
    std::vector<std::string> matrixname_;
        
  private:
    void importmatlabarray(matlabarray& ma,int mode);
    void exportmatlabarray(matlabarray& ma); 
    mitype converttype(mxtype type);
    mxtype convertclass(mlclass mclass,mitype type);
          
  public:
    // constructors
    matlabfile();
    matlabfile(std::string filename,std::string accessmode);
    virtual ~matlabfile();
  
    // open and close a file (not needed at this point)
    // access mode is "r" or "w", a combination is not supported yet
    void open(std::string filename,std::string accessmode);
    void close();

    // functions for scanning through the contents of a matlab file
    // getnummatlabarrays() gets the number of arrays stored in the file
    // and getmatlabarrayinfo() loads the matrix header but not the data
    // inside, it does read the headers of sub matrices, getmatlabarrayshortinfo()
    // on the other hand only reads the header of the top level matlabarray
    // (no submatrices are read)
    long getnummatlabarrays();

    matlabarray getmatlabarrayshortinfo(long matrixindex);
    matlabarray getmatlabarrayshortinfo(std::string name);

    matlabarray getmatlabarrayinfo(long matrixindex);
    matlabarray getmatlabarrayinfo(std::string name);
        
    // function reading matrices
    matlabarray getmatlabarray(long matrixindex);
    matlabarray getmatlabarray(std::string name);


    // function writing the matrices
    // A matrix name needs to be added. This the name of the object
    // as it appears in matlab
    void putmatlabarray(matlabarray& ma,std::string matrixname);
    
  };

} // end namespace MatlabIO

#endif
