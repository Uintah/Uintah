/*
 *   TypeName.h : template to return name of argument type; 
 *                used in PIO of templatized types
 *                
 *   Created by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *   Copyright (C) 2000 SCI Institute
 *   
 */

#include <Core/Datatypes/TypeName.h>
#include <string>

#ifndef BIOPSE_TYPENAME_H
#define BIOPSE_TYPENAME_H

namespace SCIRun {

using std::string;

class NeumannBC;
template<> string findTypeName(NeumannBC*);

} // namespace SCIRun

#endif
