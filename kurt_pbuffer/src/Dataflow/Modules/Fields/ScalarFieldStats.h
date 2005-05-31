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


//    File   : ScalarFieldStats.h
//    Author : Kurt Zimmerman
//    Date   : September 2001

#if !defined(ScalarFieldStats_h)
#define ScalarFieldStats_h

#include <Dataflow/Network/Module.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <algorithm>
#include <vector>

namespace SCIRun {
using std::sort;
using std::vector;

class ScalarFieldStats : public Module
{
  friend class ScalarFieldStatsAlgo;
public:
  ScalarFieldStats(GuiContext* ctx);
  virtual ~ScalarFieldStats();
  virtual void execute();
  
  void fill_histogram( vector<int>& hits);
  void clear_histogram();
  void local_reset_vars(){ reset_vars(); }

  GuiDouble min_;
  GuiDouble max_;
  GuiDouble mean_;
  GuiDouble median_;
  GuiDouble sigma_;   //standard deviation
  
  GuiInt is_fixed_;
  GuiInt nbuckets_;
private:

};



class ScalarFieldStatsAlgo : public DynamicAlgoBase
{
public:
  virtual void execute(FieldHandle src, ScalarFieldStats *sfs) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    const TypeDescription *lsrc);
};


template <class FIELD, class LOC>
class ScalarFieldStatsAlgoT : public ScalarFieldStatsAlgo
{
public:
  //! virtual interface. 
  virtual void execute(FieldHandle src, ScalarFieldStats *sfs);
};


template <class FIELD, class LOC>
void
ScalarFieldStatsAlgoT<FIELD, LOC>::execute(FieldHandle field_h,
					   ScalarFieldStats *sfs)
{
  //static int old_min = 0;
  //static int old_max = 0;
  static bool old_fixed = false;

  FIELD *ifield = dynamic_cast<FIELD *>(field_h.get_rep());
  typename FIELD::mesh_handle_type mesh = ifield->get_typed_mesh();

  typename LOC::iterator itr, eitr;
  mesh->begin(itr);
  mesh->end(eitr);

  bool init = false;
  typename FIELD::value_type value = 0;
  typename FIELD::value_type min = 0;
  typename FIELD::value_type max = 0;
  int counter = 0;
  vector<typename FIELD::value_type> values;

  double mean = 0;
  if( sfs->is_fixed_.get() == 1 ) {
    
    while (itr != eitr)
    {
      typename FIELD::value_type val;
      ifield->value(val, *itr);
      if( val >= sfs->min_.get() && val <= sfs->max_.get() ) {
	values.push_back( val );
	value += val;
	++counter;
      }
      ++itr;
    }
    mean = value/double(counter);
    sfs->mean_.set( mean );
  } else {
    old_fixed = false;
    while (itr != eitr)
    {
      typename FIELD::value_type val;
      ifield->value(val, *itr);
      values.push_back( val );
      value += val;
      if( !init ) {
	min = max = val;
	init = true;
      } else {
	min = (val < min) ? val:min;
	max = (val > max) ? val:max;
      }
      ++counter;
      ++itr;
    }
    mean = value/double(counter);
    sfs->mean_.set( mean );

    
    sfs->min_.set( double( min ) );
    sfs->max_.set( double( max ) );
  }

  sfs->local_reset_vars();
  if((sfs->max_.get() - sfs->min_.get()) > 1e-16 && values.size() > 0){
    int nbuckets = sfs->nbuckets_.get();
    vector<int> hits(nbuckets, 0);
    
    double frac = 1.0;
    frac = (nbuckets-1)/(sfs->max_.get() - sfs->min_.get());

    double sigma = 0.0;
    typename vector<typename FIELD::value_type>::iterator vit = values.begin();
    typename vector<typename FIELD::value_type>::iterator vit_end = values.end();
    for(; vit != vit_end; ++vit) {
      if( *vit >= sfs->min_.get() && *vit <= sfs->max_.get()){
	double value = (*vit - sfs->min_.get())*frac;
	hits[int(value)]++;
      }
      sigma += (*vit - mean)*(*vit - mean);
    }
    sfs->sigma_.set( sqrt( sigma / double(values.size()) ));
    
    vit = values.begin();
    sort(vit, vit_end);
    sfs->median_.set( double ( values[ values.size()/2] ) );
    sfs->fill_histogram( hits );
  } else {
    sfs->warning("min - max < precision or no values in range; clearing histogram");
    sfs->clear_histogram();
  }
}




} // end namespace SCIRun

#endif // ScalarFieldStats_h
