/*
 *   TypeName.h :  specializations of template<class T> 
 *                 findTypeName() function for build-in 
 *                 and simple types not deriving from Core::Datatype
 *                 
 *   Created by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   December 2000
 *
 *   Copyright (C) 2000 SCI Institute
 */

#include <string>

namespace BioPSE{
class NeumannBC;
}

#include <Packages/BioPSE/Core/Datatypes/TypeName.h>

namespace SCIRun {

using std::string;

template<> const string find_type_name(NeumannBC*)
{
  return "NeumannBC";
}

} // namespace BioPSE
