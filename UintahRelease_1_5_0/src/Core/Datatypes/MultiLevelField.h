/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#ifndef Core_Datatypes_MultiLevelField_h
#define Core_Datatypes_MultiLevelField_h

#include <Core/Datatypes/GenericField.h>

#include <vector>

namespace SCIRun {
using std::vector;

template <class FIELD>
struct MultiLevelFieldLevel 
{  
  MultiLevelFieldLevel(){};
  MultiLevelFieldLevel( vector<typename FIELD::handle_type> p, int l):
    level(l), patches(p) {}

  int level;
  vector<typename FIELD::handle_type> patches;
};

template <class Mesh, class Basis, class FData> 
class MultiLevelField : 
    public GenericField<Mesh, Basis, FData >
{
public:
  typedef GenericField<Mesh, Basis, FData> FIELD;

  MultiLevelField(){}
  MultiLevelField(vector<MultiLevelFieldLevel<FIELD>* >& levels):   
    FIELD( *((levels[0])->patches[0].get_rep()) ), levels_(levels) {}
  int nlevels() { return levels_.size(); }
  MultiLevelFieldLevel<FIELD>* level(int i) { return levels_[i]; }
  virtual ~MultiLevelField(){
   typename vector<MultiLevelFieldLevel<FIELD>* >::iterator it = levels_.begin();
   for(; it != levels_.end(); ++it){
     delete *it;
   }
  }
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(Field::td_info_e td = Field::FULL_TD_E) const;

private:
  vector<MultiLevelFieldLevel<FIELD>* > levels_;
};


template <class Mesh, class Basis, class FData>
const string
MultiLevelField<Mesh, Basis, FData>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 3);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNM
      + type_name(2) + FTNM + type_name(3) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("MultiLevelField");
    return nm;
  }
  else if (n == 1)
  {
    return find_type_name((Mesh *)0);
  }
  else if (n == 2)
  {
    return find_type_name((Basis *)0);
  }
  else
  {
    return find_type_name((FData *)0);
  }

} 

template <class Mesh, class Basis, class FData> 
const TypeDescription*
MultiLevelField<Mesh, Basis, FData>::get_type_description(Field::td_info_e td) const
{
  static string name(type_name(0));
  static string namesp("SCIRun");
  static string path(__FILE__);
  const TypeDescription *sub1 = SCIRun::get_type_description((Mesh*)0);
  const TypeDescription *sub2 = SCIRun::get_type_description((Basis*)0);
  const TypeDescription *sub3 = SCIRun::get_type_description((FData*)0);

  switch (td) {
  default:
  case Field::FULL_TD_E:
    {
      static TypeDescription* tdn1 = 0;
      if (tdn1 == 0) {
	TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(3);
	(*subs)[0] = sub1;
	(*subs)[1] = sub2;
	(*subs)[2] = sub3;
	tdn1 = scinew TypeDescription(name, subs, path, namesp, 
				      TypeDescription::FIELD_E);
      } 
      return tdn1;
    }
  case Field::FIELD_NAME_ONLY_E:
    {
      static TypeDescription* tdn0 = 0;
      if (tdn0 == 0) {
	tdn0 = scinew TypeDescription(name, 0, path, namesp, 
				      TypeDescription::FIELD_E);
      }
      return tdn0;
    }
  case Field::MESH_TD_E:
    {
      return sub1;
    }
  case Field::BASIS_TD_E:
    {
      return sub2;
    }
  case Field::FDATA_TD_E:
    {
      return sub3;
    }
  };
}

} // end namespace SCIRun

#endif
