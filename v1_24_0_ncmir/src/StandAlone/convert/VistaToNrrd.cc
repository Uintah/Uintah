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
 *  VistaToNrrd.cc: create a nrrd header for a vista .v file
 *
 *  Written by:
 *   Mark Hartner
 *   SCI Institute
 *   University of Utah
 *   December 2004
 *
 *  Copyright (C) 2004 SCI Group
 */

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

using namespace std;

int main(int argc, char **argv){

  int nrrdSkip = 0;
  char *in = argv[1];
  ifstream vistaFileStream(in, ios::binary);
  char *out = argv[2];
  ofstream nrrdHeader(out, ios::binary);
  string z_dim("0");
  string y_dim("0");
  string x_dim("0");
  
  if (argc < 3) {
    cerr << "Usage: "<<argv[0]<<" inputFile.v outputFile.nhdr \n";
    return 2;
  }

	
  if (! vistaFileStream){
    cout<<"could not input vista file "<<in<<endl;
    return 1;
  }
  if (! nrrdHeader){
    cout<<"could not create nrrd header file "<<out<<endl;
    return 1;
  }
	
  int lineCount = 0;
  int foundFormFeed = 0;	
  while (! vistaFileStream.eof() && !foundFormFeed){
    string current_line;
    getline(vistaFileStream,current_line);
    lineCount++;

    if (current_line[0] == '\f'){
      nrrdSkip = lineCount;
      foundFormFeed = 1;
    }

    {
      //look for the Z factor line because we need that for NRRD spacing
      string z_marker("nframes: ");
      string::size_type z_factor_pos = current_line.find(z_marker);
      if ( z_factor_pos != string::npos ){
	z_dim = current_line.substr(z_factor_pos + z_marker.size());
      }
    }
    
    //look for the Y factor line because we need that for NRRD spacing
    {
      string y_marker("nrows: ");
      string::size_type y_factor_pos = current_line.find(y_marker);
      if ( y_factor_pos != string::npos ){
	y_dim = current_line.substr(y_factor_pos + y_marker.size());
      }
    }

    //look for the X factor line because we need that for NRRD spacing
    {
      string x_marker("ncolumns: ");
      string::size_type x_factor_pos = current_line.find(x_marker);
      if ( x_factor_pos != string::npos ){
	x_dim = current_line.substr(x_factor_pos + x_marker.size());
      }
    }
    


  }
  vistaFileStream.close();

  nrrdHeader<<"NRRD0001"<<endl
	    <<"type: uchar"<<endl
	    <<"dimension: 3"<<endl
	    <<"spacings: 1 1 1"<<endl
	    <<"sizes: "<<x_dim<<" "<<y_dim<<" "<<z_dim<<endl
	    <<"data file: "<<in<<endl
	    <<"endian: big"<<endl
	    <<"encoding: raw"<<endl
	    <<"line skip: "<<nrrdSkip<<endl;

  nrrdHeader.close();
  return 0;
}
   
