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
 *  strauss.h
 *                   
 *
 *  Written by:
 *   Kostadin Damevski
 *   School of Computing
 *   University of Utah
 *   October, 2003
 *
 *  Copyright (C) 2003 SCI 
 */

#ifndef strauss_h
#define strauss_h

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

//CRCState is used a container to emit code into
//It calculates a crc for all strings written
struct Leader {
};

class CRCState : public std::ostringstream {
 public:
  CRCState::CRCState() {
    //Populate the crc table
    unsigned long poly = 0xEDB88320L;
    unsigned long entry;
    int i, j;
   
    for(i=0; i<256; i++) {
      entry = i;
      for(j=8; j>0; j--) {
	if(entry & 1)
	  entry = (entry >> 1) ^ poly;
	else
	  entry >>= 1;
      }
      crc_table[i] = entry;
    }
    //Init crc
    crc = 0xFFFFFFFF;
  }
  CRCState::~CRCState() {}

  std::string push_leader() {
    string oldleader=leader;
    leader+="  ";
    return oldleader;
  }
  void pop_leader(const std::string& oldleader) {
    leader=oldleader;
  }

  friend std::ostream& operator<<(ostream& out, const Leader&);

  //Calculate CRC of current text
  //omit specifies a string to skip in calculation 
  unsigned long calcCRC(std::string omit)    
  {
    string text = this->str();
    unsigned int oi = text.find(omit);
    std::cerr << "calcCRC text.size()=" << text.size() << "\n";
    
    for(unsigned int i=0;i<text.size();i++) {
      while((i == oi)&&(oi > 0)) {
        i += omit.size();
        oi = text.find(omit,i);
      } 
      crc = (((crc)>>8) & 0x00FFFFFF) ^ crc_table[ ((crc)^(int)(text[i])) & 0xFF ];
    }
    crc = ((crc)^0xFFFFFFFF);
    return crc;
  }

  std::string leader;
  unsigned long crc_table[256];
  unsigned long crc;
};

std::ostream& operator<<(CRCState& out, const Leader&);
//**** End of CRCState

namespace SCIRun {
  class Strauss {
  public:
    Strauss(string plugin, string hdrplugin, string portspec, 
	    string header, string implementation, 
	    string util, string templateArgv = "");
    ~Strauss();

    /////
    // Emits code into CRCState classes for header and implementation;
    // Returns nonzero on encountered error
    int emit();

    /////
    // If emitted returns a string of the implementation/header
    unsigned long getImplCRC();
    unsigned long getHdrCRC();

  private:
    /////
    // Commits emitted data into output files 
    void commitToFiles();	    

    Strauss();    

    ///////
    // Filenames of output files
    string header;
    string implementation;
    string plugin;
    string hdrplugin;
    string portspec;
    string util; 
    string templateArgv;

    /////
    // Randomly generated name for the generated component
    string bridgeComponent;

    ////
    // Output containing classes
    CRCState hdr;
    CRCState impl;

    ////
    // Has the code been generated
    bool emitted; 
  };
} //End of SCIRun namespace

#endif




