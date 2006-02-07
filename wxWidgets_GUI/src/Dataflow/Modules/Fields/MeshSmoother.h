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


//    File   : MeshSmoother.h
//    Author : Jason Shepherd
//    Date   : January 2006

#if !defined(MeshSmoother_h)
#define MeshSmoother_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <sci_hash_map.h>
#include <algorithm>
#include <set>

namespace SCIRun {

using std::copy;

class GuiInterface;

class MeshSmootherAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};


template <class FIELD>
class MeshSmootherAlgoTet : public MeshSmootherAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshSmootherAlgoTet<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  return field;
}

template <class FIELD>
class MeshSmootherAlgoHex : public MeshSmootherAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshSmootherAlgoHex<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
   FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
   return field;
}


} // end namespace SCIRun

#endif // MeshSmoother_h
