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

#ifndef Uintah_Datatypes_LatVolField_h
#define Uintah_Datatypes_LatVolField_h


#include <Core/Datatypes/LatVolField.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
using std::vector;


template <class Data>
struct MultiResLevel 
{  
  MultiResLevel(){};
  MultiResLevel( vector<LockingHandle<LatVolField<Data> > >p, int l):
    level(l), patches(p) {}

  int level;
  vector<LockingHandle<LatVolField<Data> > > patches;
};

template <class Data> 
class MRLatVolField : public LatVolField<Data>
{
public:
  MRLatVolField(){}
  MRLatVolField(vector<MultiResLevel<Data>* >& levels):   
    LatVolField<Data>( *((levels[0])->patches[0].get_rep()) ), levels_(levels) {}
  int nlevels() { return levels_.size(); }
  MultiResLevel<Data>* level(int i) { return levels_[i]; }
  virtual ~MRLatVolField(){}
  static const string type_name(int n = -1);
  virtual const TypeDescription* get_type_description(int n = -1) const;

private:
  vector<MultiResLevel<Data>* > levels_;
};


//  template <class Data>
//  MRLatVolField<Data>::~MRLatVolField()
//  {
//    typename vector<MultiResLevel<Data>*>::iterator it = levels_.begin();
//    typename vector<MultiResLevel<Data>*>::iterator it_end = levels_.end();

//    for(; it != it_end; ++it ){
//      typename vector<LockingHandle<LatVolField<Data> > >::iterator jt = (*it)->patches.begin();
//      typename vector<LockingHandle<LatVolField<Data> > >::iterator jt_end = (*it)->patches.end();

//      for(; jt != jt_end; ++jt )
//        delete (*jt);
//    }
//  }

template <class Data>
const string
MRLatVolField<Data>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "MRLatVolField";
  }
  else
  {
    return find_type_name((Data *)0);
  }
} 

template <class Data> 
const TypeDescription*
MRLatVolField<Data>::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("SCIRun");
  static string path(__FILE__);

  if (n == -1) {
    static TypeDescription* tdn1 = 0;
    if (tdn1 == 0) {
      const TypeDescription *sub = SCIRun::get_type_description((Data*)0);
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
      tdnn = (TypeDescription *) SCIRun::get_type_description((Data*)0);
    }
    td = tdnn;
  }
  return td;
}

} // end namespace SCIRun

#endif
