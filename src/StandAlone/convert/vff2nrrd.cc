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
 *  vff2nrrd.cc: create a nrrd header for a vff file
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
  int x=0, y=0, z=0;
  char *in = argv[1];
  ifstream vffFileStream(in, ios::binary);
  char *out = argv[2];
  ofstream nrrdHeader(out, ios::binary);

  if (argc < 3) {
    cerr << "Usage: "<<argv[0]<<" inputFile.vff outputFile.nhdr \n";
    return 2;
  }

	
  if (! vffFileStream){
    cout<<"could not input vff file "<<in<<endl;
    return 1;
  }
  if (! nrrdHeader){
    cout<<"could not create nrrd header file "<<out<<endl;
    return 1;
  }
	
  int lineCount = 0;
  int foundFormFeed = 0;	
  char temp[1025];
  while (! vffFileStream.eof() && !foundFormFeed){
    vffFileStream.getline(temp,1024);
    lineCount++;

    if (temp[0] == '\f'){
      nrrdSkip = lineCount;
      foundFormFeed = 1;
    }
	  
    if (strncmp(temp,"size=",5) == 0){
      istringstream sizeLine(&temp[5]);
      sizeLine>>x>>y>>z;	    
    }

  }
  vffFileStream.close();

  nrrdHeader<<"NRRD0001"<<endl
	    <<"type: short"<<endl
	    <<"dimension: 3"<<endl
	    <<"sizes: "<<x<<" "<<y<<" "<<z<<endl
	    <<"spacings: 1 1 1"<<endl
	    <<"data file: "<<in<<endl
	    <<"endian: big"<<endl
	    <<"encoding: raw"<<endl
	    <<"line skip: "<<nrrdSkip<<endl;

  nrrdHeader.close();
  return 0;
}

