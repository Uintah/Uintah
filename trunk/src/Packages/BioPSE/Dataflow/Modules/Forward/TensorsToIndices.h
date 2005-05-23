#ifndef SCIRun_TensorsToIndices_H
#define SCIRun_TensorsToIndices_H
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


#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Geometry/Tensor.h>
#include <sci_hash_map.h>


namespace SCIRun {

// just get the first point from the field
class TensorsToIndicesAlgo : public DynamicAlgoBase
{
public:
  virtual FieldHandle execute(FieldHandle srcH) = 0;

  //! support the dynamically compiled algorithm concept
 static CompileInfoHandle get_compile_info(const TypeDescription *field_src_td,
				           const string &field_dst_name);

#ifdef HAVE_HASH_MAP
  struct tensorhash
  {
    size_t operator()(const Tensor &t) const
    {
      unsigned char *s = (unsigned char *)(t.mat_);
      size_t h = 0;
      for (unsigned int i = 0; i < sizeof(double)*9; i++)
      {
        h = ( h << 5 ) - h + s[i];
      }
      return h;
    }
  };
  struct tensorequal
  {
    bool operator()(const Tensor &a, const Tensor &b) const
    {
      return (a == b);
    }
  };
  typedef hash_map<Tensor, int, tensorhash, tensorequal> tensor_map_type;
#else
  // TODO: map is not a replacement for hash_map.  It's not-found
  // semantics are different.  No equal test.  Just implement our own
  // hash_map class if it's not there.
  struct tensorless
  {
    bool operator()(const Tensor &a, const Tensor &b) const
    {
      for(int i=0;i<3;i++)
        for(int j=0;j<3;j++)
          if( a.mat_[i][j] >= b.mat_[i][j])
            return false;
      return true;
    }
  };
  typedef map<Tensor, int, tensorless> tensor_map_type;
#endif
};


template <class FSRC, class FDST>
class TensorsToIndicesAlgoT : public TensorsToIndicesAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(FieldHandle srcH);
};


template <class FSRC, class FDST>
FieldHandle
TensorsToIndicesAlgoT<FSRC, FDST>::execute(FieldHandle srcH)
{
  FSRC *src = dynamic_cast<FSRC *>(srcH.get_rep());
  FDST *dst = scinew FDST(src->get_typed_mesh(), src->basis_order());
  vector<pair<string, Tensor> > conds;
  typename FSRC::fdata_type::iterator in = src->fdata().begin();
  typename FDST::fdata_type::iterator out = dst->fdata().begin();
  typename FSRC::fdata_type::iterator end = src->fdata().end();

  tensor_map_type tmap;

  while (in != end)
  {
    const typename tensor_map_type::iterator loc = tmap.find(*in);
    if (loc != tmap.end())
    {
      *out = (*loc).second;
    }
    else
    {
      conds.push_back(pair<string, Tensor>("x", *in));
      tmap[*in] = conds.size() - 1;
      *out = conds.size() - 1;
    }

    ++in; ++out;
  }
  dst->set_property("conductivity_table", conds, false);
  return dst;
}
} // End namespace BioPSE

#endif
