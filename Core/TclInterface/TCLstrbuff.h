/*
 *  TCLstrbuff.h: class to define string buffer for TclStream itcl-class 
 * 
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   January, 2001
 *   
 *   Copyright (C) 2000 SCI Group
 */

#ifndef TCL_OSTREAM_H
#define TCL_OSTREAM_H

#include <Core/TclInterface/TCLvar.h>

#include <sstream>
#include <iostream>

namespace SCIRun {

using namespace std;

class SCICORESHARE TCLstrbuff : public TCLvar, public ostringstream{
  
  // GROUP: private data
  //////////
  // Placeholder to C-type string for communication with Tcl
  char*      d_buff;
  //////////
  // Size of allocated C-string
  int        d_bSize;

public:
  
  // GROUP: Constructor/Destructor
  //////////
  // 
  TCLstrbuff(const clString& name, const clString& id, TCL* tcl);
  virtual ~TCLstrbuff();
  
  // GROUP: public member functions
  //////////
  // 
  TCLstrbuff& flush();
 
  template<class T> inline TCLstrbuff& operator<<(T pVal){
    static_cast<ostringstream&>(*this)<<pVal;
    return *this;
  }

  virtual void emit(std::ostream&);
};

TCLstrbuff& operator<<(TCLstrbuff& stream, TCLstrbuff& (*mp)(TCLstrbuff&));
TCLstrbuff& endl(TCLstrbuff& stream);
TCLstrbuff& flush(TCLstrbuff& stream);
TCLstrbuff& ends(TCLstrbuff& stream);

} // end namespace SCIRun

#endif
