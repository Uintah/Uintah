/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

#include <fstream>
#include <vector>
#include <sstream>

#include <Core/CCA/tools/strauss/c++ruby/rubyeval.h>
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
  unsigned long calcCRC(std::string& omit)    
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
    Strauss(string plugin, string portspec, 
	    string header, string implementation);
    ~Strauss();

    /////
    // Emits code into CRCState classes for header and implementation;
    // Returns nonzero on encountered error
    int emit();

    /////
    // If emitted returns a string of the implementation/header
    unsigned long getImplCRC();
    unsigned long getHdrCRC();

    /////
    // Commits emitted data into output files 
    void commitToFiles();	    

  private:
    Strauss();    

    ///////
    // Filenames of output files
    string header;
    string implementation;
    string plugin;
    string portSpec;

    //////////
    // Collection of file streams that we emit bridge into.
    ofstream fHeader;
    ofstream fImpl;

    //////
    // Ruby expression evaluating class   
    RubyEval* ruby; 

    /////
    // Bridge Component Name (generated here)
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




