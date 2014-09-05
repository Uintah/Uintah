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

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Containers/Array1.h>
#include <vector>

namespace BioPSE {

using namespace SCIRun;

class SegLatVolField : public LatVolField<int> {
private:
  int maxMatl_;
  Array1<pair<int, long> > comps_;
  Array1<Array1<LatVolMesh::Cell::index_type> *> compMembers_;
  int lowestValue(int i, Array1<int>& workingLabels);
  void setAll(int i, int newValue, Array1<int>& workingLabels);
  void setEquiv(int larger, int smaller, Array1<int>& workingLabels);
  void initialize();

public:
  SegLatVolField() : LatVolField<int>() {}
  SegLatVolField(const SegLatVolField &copy);
  SegLatVolField(LatVolMeshHandle mesh)
    : LatVolField<int>(mesh, Field::CELL)
  {
  }
  
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
