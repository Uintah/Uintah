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

//    File   : LinAlgUnary.h
//    Author : Michael Callahan
//    Date   : June 2002

#if !defined(LinAlgUnary_h)
#define LinAlgUnary_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>

namespace SCIRun {

class LinAlgUnaryAlgo : public DynamicAlgoBase
{
public:
  virtual double user_function(double x) = 0;

  virtual string identify() = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const string &function,
					    int hashoffset);
};


} // end namespace SCIRun

#endif // LinAlgUnary_h
