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
 * FILE: timedatafile.h
 * AUTH: Jeroen G Stinstra
 * DATE: 23 MAR 2005
 */
 
// STILL TO DO
// (1) ADD SPACING / KINDS / UNITS INFORMATION
// (2) DEAL WITH 3D DATA AS WELL
// (3) UPGRADE ROW/COLUMN SECTION 
 
 
#ifndef JGS_CARDIOWAVE_TIMEDATATILE_H
#define JGS_CARDIOWAVE_TIMEDATATILE_H 1

#include <Core/Thread/Time.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/NrrdString.h>
#include <Core/Exceptions/Exception.h>


#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <list>
#include <sgi_stl_warnings_on.h>


using namespace SCIRun;
using namespace std;

namespace CardioWave {

	class TimeDataFileException : public Exception {
	public:
	    TimeDataFileException(const std::string&);
	    TimeDataFileException(const TimeDataFileException&);
	    virtual ~TimeDataFileException();
	    virtual const char* message() const;
	    virtual const char* type() const;
	protected:
	private:
	    std::string message_;
	    TimeDataFileException& operator=(const TimeDataFileException&);
	};


class TimeDataFile {

  public:
    // Constructor
    TimeDataFile();
    TimeDataFile(std::string filename);
    
    // Destructor
    virtual ~TimeDataFile();
    
    // open function
    void open(std::string filename);  // Open the datafile

    // Get the essential Nrrd Header data
    int         getsize(int dim);     // Get the number of samples in dimension dim
    double      getspacing(int dim);  // Get the spacing in dimension dim
    std::string getkind(int dim);     // Get the kind of the axis 
    std::string getunit(int dim);     // Get the unit along dimension dim
    std::string getcontent();         // Get the content field of the nrrd
    std::string getsampleunit();      // Get the unit of the data.

    bool getmatrix(MatrixHandle& mh,int dim, int dimstart, int dimend);
    
  private:  


    std::string remspaces(std::string str); // Remove white spaces from data read
    void gettype(std::string type,int& nrrdtype, int& elsize);  // Get the nrrdtype and the element size
    int  cmp_nocase(const std::string& s1, const std::string& s2);  // Nrrd header is text, make it case insensitive
    void doswapbytes(void *vbuffer,long elsize,long size); // Swap bytes if needed
  
    std::string               datafilename_;    // if there is one file use this one
    std::vector<std::string>  datafilenames_;   // if data is spread over multiple files use this one
    std::vector<std::string>  units_;           // Unit per axis
    std::vector<std::string>  kinds_;           // Kind per axis
    std::vector<double>       spacings_;        // Spacing per axis
    std::vector<int>          sizes_;           // Size per axis
    std::vector<int>          sdoffset_;        // Internal use only
    int                       sdsize_;          

    std::string             content_;
    std::string             encoding_;
    std::string             endian_;
    std::string             type_;
    std::string             unit_;
    int                     lineskip_;
    int                     byteskip_;
    int                     elemsize_;
    int                     ntype_;
    int                     dimension_;

    std::map<std::string,std::string> keyvalue_;
    
    int                     start_;
    int                     end_;
    int                     step_;
    int                     subdim_;
    
    bool                    useformatting_;
    bool                    swapbytes_;
    
  };
  
}

#endif
