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
 *  MaskedHexVolField.h
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2001 SCI Institute
 */

#ifndef Datatypes_MaskedHexVolField_h
#define Datatypes_MaskedHexVolField_h

#include <Core/Datatypes/HexVolField.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template <class T> 
class MaskedHexVolField : public HexVolField<T> {
private:
  vector<char> mask_;  // since Pio isn't implemented for bool's
public:
  vector<char>& mask() { return mask_; }

  MaskedHexVolField() :
    HexVolField<T>() {};
  MaskedHexVolField(int order) : 
    HexVolField<T>(order) {};
  MaskedHexVolField(HexVolMeshHandle mesh, int order) : 
    HexVolField<T>(mesh, order) 
  {
    resize_fdata();
  };

  bool get_valid_nodes_and_data(vector<pair<HexVolMesh::Node::index_type, T> > &data) {
    data.erase(data.begin(), data.end());
    if (this->basis_order() != 1) return false;
    HexVolMesh::Node::iterator ni, nie;
    this->mesh_->begin(ni);
    this->mesh_->end(nie);
    for (; ni != nie; ++ni) { 
      if (mask_[*ni]) { 
	pair<HexVolMesh::Node::index_type, T> p;
	p.first=*ni; 
	p.second=this->fdata()[*ni];
	data.push_back(p);
      }
    }
    return true;
  }

  virtual ~MaskedHexVolField() {};

  bool value(T &val, HexVolMesh::Node::index_type i) const
  { if (!mask_[i]) return false; val = this->fdata()[i]; return true; }
  bool value(T &val, HexVolMesh::Edge::index_type i) const
  { if (!mask_[i]) return false; val = this->fdata()[i]; return true; }
  bool value(T &val, HexVolMesh::Face::index_type i) const
  { if (!mask_[i]) return false; val = this->fdata()[i]; return true; }
  bool value(T &val, HexVolMesh::Cell::index_type i) const
  { if (!mask_[i]) return false; val = this->fdata()[i]; return true; }

  void initialize_mask(char masked) {
    for (vector<char>::iterator c = mask_.begin(); c != mask_.end(); ++c) *c=masked;
  }

  // Have to be explicit about where mesh_type comes from for IBM xlC
  // compiler... is there a better way to do this?
  typedef GenericField<HexVolMesh,vector<T> > GF;

  void resize_fdata() {
    if (this->basis_order() == 0)
    {
      typename GF::mesh_type::Cell::size_type ssize;
      this->mesh_->size(ssize);
      mask_.resize(ssize);
    }
    else
    {
      typename GF::mesh_type::Node::size_type ssize;
      this->mesh_->size(ssize);
      mask_.resize(ssize);
    }
    HexVolField<T>::resize_fdata();
  }

  //! Persistent IO
  static PersistentTypeID type_id;
  virtual void io(Piostream &stream);
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  static Persistent *maker();
};

// Pio defs.
const int MASKED_HEX_VOL_FIELD_VERSION = 1;

template <class T>
Persistent*
MaskedHexVolField<T>::maker()
{
  return scinew MaskedHexVolField<T>;
}

template <class T>
PersistentTypeID 
MaskedHexVolField<T>::type_id(type_name(-1), 
			 HexVolField<T>::type_name(-1),
			 maker);


template <class T>
void 
MaskedHexVolField<T>::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), 
				     MASKED_HEX_VOL_FIELD_VERSION);
  HexVolField<T>::io(stream);
  Pio(stream, mask_);
  stream.end_class();
}

template <class T> 
const string 
MaskedHexVolField<T>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MaskedHexVolField";
  }
  else
  {
    return find_type_name((T *)0);
  }
}

template <class T> 
const TypeDescription*
MaskedHexVolField<T>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if (n == -1) {
    static TypeDescription* tdn1 = 0;
    if (tdn1 == 0) {
      const TypeDescription *sub = SCIRun::get_type_description((T*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      tdn1 = scinew TypeDescription(name, subs, path, namesp);
    } 
    td = tdn1;
  }
  else if(n == 0) {
    static TypeDescription* tdn0 = 0;
    if (tdn0 == 0) {
      tdn0 = scinew TypeDescription(name, 0, path, namesp);
    }
    td = tdn0;
  }
  else {
    static TypeDescription* tdnn = 0;
    if (tdnn == 0) {
      tdnn = (TypeDescription *) SCIRun::get_type_description((T*)0);
    }
    td = tdnn;
  }
  return td;
}

} // end namespace SCIRun

#endif // Datatypes_MaskedHexVolField_h
