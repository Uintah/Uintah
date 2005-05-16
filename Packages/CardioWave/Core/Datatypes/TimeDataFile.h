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
    void open(std::string filename);
    
    int getncols();
    int getnrows();
    
    std::string getcontent();
    std::string getunit();
    
    void getcolmatrix(MatrixHandle& mh,int colstart,int colend);
    void getrowmatrix(MatrixHandle& mh,int rowstart,int rowend);
    void getcolnrrd(NrrdDataHandle& nh,int colstart,int colend);
    void getrownrrd(NrrdDataHandle& nh,int rowstart,int rowend);

//    void getcolmatrix(MatrixHandle& mh,std::vector<int> indices);
//    void getrowmatrix(MatrixHandle& mh,std::vector<int> indices);
//    void getcolnrrd(NrrdDataHandle& nh,std::vector<int> indices);
//    void getrownrrd(NrrdDataHandle& nh,std::vector<int> indices);

    
  private:  
    std::string remspaces(std::string str);
    void gettype(std::string type,int& nrrdtype, int& elsize);
    int  cmp_nocase(const std::string& s1, const std::string& s2);
    void doswapbytes(void *vbuffer,long elsize,long size);
  
  private:
  
    std::string datafilename;
    std::list<std::string> datafilenames;
    std::string content;
    std::string encoding;
    std::string endian;
    std::string type;
    std::string unit;
    int         ncols;
    int         nrows;
    int         ndims;
    int         lineskip;
    int         byteskip;
    int         elemsize;
    int         ntype;
    int         dimension;
    std::map<std::string,std::string> keyvalue;
    
    int         start;
    int         end;
    int         step;
    int         subdim;
    bool        useformatting;
    
    
    bool        swapbytes;
    
    FILE*        datafile;
    int          datafile_uni;
    
  };
  
}

#endif
