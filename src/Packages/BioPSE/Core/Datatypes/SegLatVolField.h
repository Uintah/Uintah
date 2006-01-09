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
 *  SegLatVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_SegLatVolField_h
#define Datatypes_SegLatVolField_h

#include <Core/Geometry/Point.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Persistent/PersistentSTL.h>
#include <vector>

namespace SCIRun {
void Pio(Piostream &stream, 
	 LatVolMesh<HexTrilinearLgn<Point> >::CellIndex &n);

}
#include <Packages/BioPSE/Core/Datatypes/share.h>

namespace BioPSE {

using namespace SCIRun;

typedef LatVolMesh<HexTrilinearLgn<Point> >  SegLVMesh;
typedef ConstantBasis<int>                 SegDatBasis;
typedef GenericField<SegLVMesh, SegDatBasis,
		     FData3d<int, SegLVMesh> > SegLVField;

class SHARE SegLatVolField : public SegLVField {
private:
  int maxMatl_;
  Array1<pair<int, long> > comps_;
  Array1<Array1<SegLVMesh::Cell::index_type> *> compMembers_;
  int lowestValue(int i, Array1<int>& workingLabels);
  void setAll(int i, int newValue, Array1<int>& workingLabels);
  void setEquiv(int larger, int smaller, Array1<int>& workingLabels);
  void initialize();

public:
  SegLatVolField() : SegLVField() {}
  SegLatVolField(const SegLatVolField &copy);
  SegLatVolField(SegLVMesh::handle_type mesh) : 
    SegLVField(mesh)
  {}
  
  inline int ncomps() { return comps_.size(); }
  inline int compMatl(int comp) { return comps_[comp].first; }
  inline long compSize(int comp) { return comps_[comp].second; }

  virtual SegLatVolField *clone() const;
  virtual ~SegLatVolField() {};

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);

  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

  void setData(const Array3<int> &fdata);
  void audit();
  void printComponents();
  void compress();
  void absorbComponent(int old_comp, int new_comp);
  void absorbSmallComponents(int min);
  
private:
  static Persistent *maker();
};

} // end namespace BioPSE

#endif // Datatypes_SegLatVolField_h
