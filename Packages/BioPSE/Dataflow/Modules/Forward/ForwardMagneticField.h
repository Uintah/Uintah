//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : ForwardMagneticField.h
//    Author : Robert Van Uitert
//    Date   : Mon Aug  4 14:46:51 2003

#include <Dataflow/Network/Module.h>
#include <Core/Thread/Thread.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Modules/Fields/ChangeFieldDataType.h>
#include <Core/Thread/Mutex.h>
#include <sci_hash_map.h>

namespace BioPSE {

using namespace SCIRun;


class ForwardMagneticField : public Module {
public:
  ForwardMagneticField(GuiContext *context);

  virtual ~ForwardMagneticField();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
private:
  FieldIPort* electricFieldP_;
  FieldIPort* cond_tens_;
  FieldIPort* sourceLocationP_;
  FieldIPort* detectorPtsP_;
  FieldOPort* magneticFieldAtPointsP_;
  FieldOPort* magnitudeFieldP_;
};

class CalcFMFieldBase : public DynamicAlgoBase
{
public:
  virtual bool calc_forward_magnetic_field(FieldHandle efield, 
					   FieldHandle ctfield,
					   FieldHandle dipoles, 
					   FieldHandle detectors, 
					   FieldHandle &magnetic_field, 
					   FieldHandle &magnitudes,
					   int np,
					   ProgressReporter *mod) = 0;

  virtual ~CalcFMFieldBase();
  
  static const string& get_h_file_path();
  static const string base_class_name() {
    static string name("CalcFMFieldBase");
    return name;
  }

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *efld_td, 
					    const TypeDescription *ctfld_td,
					    const TypeDescription *detfld_td);
};

template <class ElecField, class CondField, class PointField>
class CalcFMField : public CalcFMFieldBase
{
public:

  CalcFMField() :
    np_(-1)
  {}
  
  

  //! virtual interface.
  virtual bool calc_forward_magnetic_field(FieldHandle efield, 
					   FieldHandle ctfield,
					   FieldHandle dipoles, 
					   FieldHandle detectors, 
					   FieldHandle &magnetic_field, 
					   FieldHandle &magnitudes,
					   int np,
					   ProgressReporter *mod);

private:
  typedef CalcFMField<ElecField, CondField, PointField> AlgoType;
  typedef typename ElecField::mesh_type::Cell::iterator EFieldCIter;
  typedef typename ElecField::mesh_type::Cell::iterator EFieldCIndex;
  typedef typename PointField::mesh_type::Node::iterator PFieldNIter;
  typedef typename PointField::mesh_type::Node::index_type PFieldNIndex;

  void interpolate(int proc, Point p);
  void set_up_cell_cache();
  void calc_parallel(int proc, ProgressReporter *mod);
  void get_parallel_iter(int np);
  void set_parallel_data(PointField *fout, FieldHandle &magnitudes,  
			 Handle<ChangeFieldDataTypeAlgoCreate> create_algo);

  int                                                     np_;
  vector<Vector>                                          interp_value_;

  struct per_cell_cache {
    Vector cur_density_;
    Point  center_;
    double volume_;
  };
  
  typedef hash_map<unsigned, per_cell_cache, hash<unsigned> > cache_t;
  cache_t                                                  cell_cache_;

  ElecField                                               *efld_;
  CondField                                               *ctfld_;
  PointField                                              *dipfld_;
  PointField                                              *detfld_;
  vector<pair<string, Tensor> >                            tens_;
  vector<pair<PFieldNIter, PFieldNIter> >                  piters_;
  vector<vector<pair<double, Vector> > >                   data_out_;
};


template <class ElecField, class CondField, class PointField>
void
CalcFMField<ElecField, CondField, PointField>::calc_parallel(int proc,
							 ProgressReporter *mod)
{
  PFieldNIter iter, end;
  pair<PFieldNIter, PFieldNIter> iters = piters_[proc];

  iter = iters.first;
  end = iters.second;

  typename PointField::mesh_handle_type mesh = detfld_->get_typed_mesh();
  typename PointField::mesh_handle_type dip_mesh = dipfld_->get_typed_mesh();

  // iterate over the detectors.
  while (iter != end) {
    // finish loop iteration.

    PFieldNIndex ind = *iter;
    ++iter;
    //    mod->update_progress(count_, det_sz);    
    Vector mag_field;
    Point  pt;
    mesh->get_center(pt, ind);


    // init the interp val to 0 
    interp_value_[proc] = Vector(0,0,0);
   
    interpolate(proc, pt);

    mag_field = interp_value_[proc];

    Vector normal;
    normal = Point(0,0,0) - pt;//detfld->value(ind);
    // start of B(P) stuff

    PFieldNIter dip_iter, dip_end;
    dip_mesh->begin(dip_iter);
    dip_mesh->end(dip_end);   

    // iterate over the dipoles.
    while (dip_iter != dip_end) {

      PFieldNIndex dip_ind = *dip_iter;
      ++dip_iter;

      Point  pt2;
      dip_mesh->get_center(pt2, dip_ind);
      Vector P   = dipfld_->value(dip_ind);
      
      Vector radius = pt - pt2; // detector - source

      Vector valuePXR = Cross(P, radius);
      double length = radius.length();
 
      mag_field += valuePXR / (length * length * length);
      
    }
    // end of B(P) stuff
    const double one_over_4_pi = 1.0 / (4 * M_PI);
    pair<double, Vector> p;
    p.first = Dot(mag_field, normal);
    mag_field *= one_over_4_pi;
    p.second  = mag_field;

    //cout <<  "scalar: " << p.first << endl;
    //cout << "vector: " << p.second << endl;
    data_out_[proc].push_back(p);
  }
}

//! Assume that the value_type for the input fields are Vector.
template <class ElecField, class CondField, class PointField>
bool
CalcFMField<ElecField, CondField, PointField>::calc_forward_magnetic_field(
      					    FieldHandle efield, 
					    FieldHandle ctfield, 
					    FieldHandle dipoles, 
					    FieldHandle detectors, 
					    FieldHandle &magnetic_field,
					    FieldHandle &magnitudes,
					    int np,
					    ProgressReporter *mod)
{
  ElecField *efld = dynamic_cast<ElecField*>(efield.get_rep());
  ASSERT(efld != 0);
  efld_ = efld;

  CondField *ctfld = dynamic_cast<CondField*>(ctfield.get_rep());
  ASSERT(ctfld != 0);
  ctfld_ = ctfld;

  if (! ctfld_->get_property("conductivity_table", tens_)) {
    mod->error("Must have a conductivity table in Conductivity Field");
    return false;
  }

  PointField *dipfld = dynamic_cast<PointField*>(dipoles.get_rep());
  ASSERT(dipfld != 0);
  dipfld_ = dipfld;

  PointField *detfld = dynamic_cast<PointField*>(detectors.get_rep());
  ASSERT(detfld != 0);
  detfld_ = detfld;

  // create the output fields
  PointField *fout = scinew PointField(detfld->get_typed_mesh(), 
				       detfld->data_at());
  magnetic_field = fout;

  // The magnitude field has a shared mesh wity detectors, but is scalar.
  const string new_field_type =
    detectors->get_type_description(0)->get_name() + "<double> ";
  // Create a field identical to the input, except for the edits.
  const TypeDescription *fsrc_td = detectors->get_type_description();
  CompileInfoHandle create_ci =
    ChangeFieldDataTypeAlgoCreate::get_compile_info(fsrc_td, new_field_type);
  Handle<ChangeFieldDataTypeAlgoCreate> create_algo;
  if (!DynamicCompilation::compile(create_ci, create_algo, this))
  {
    mod->error("Unable to compile creation algorithm.");
    return false;
  }
  
  magnitudes = create_algo->execute(detectors);
  
  typedef typename PointField::value_type val_t;
  typename PointField::mesh_handle_type mesh = detfld->get_typed_mesh();
  typename PointField::mesh_handle_type dip_mesh = dipfld->get_typed_mesh();
  mesh->synchronize(Mesh::NODES_E);

  // init parallel iterators
  get_parallel_iter(np);
  
  // cache per cell calculations that are used over and over again.
  set_up_cell_cache();
  
  data_out_.resize(np_);
  // do the parallel work.
  Thread::parallel(Parallel1<AlgoType, ProgressReporter*>(this, 
						     &AlgoType::calc_parallel,
						      mod), np_, true);
  
  //iterate over output fields and set the values...
  set_parallel_data(fout, magnitudes,  create_algo);
  return true;
}

template <class ElecField, class CondField, class PointField>
void
CalcFMField<ElecField, CondField, PointField>::set_parallel_data(
		      PointField *fout, FieldHandle &magnitudes,  
		      Handle<ChangeFieldDataTypeAlgoCreate> create_algo)
{
  int chunk = 0;
  vector<vector<pair<double, Vector> > >::iterator iter = data_out_.begin();
  while (iter != data_out_.end()) {
    vector<pair<double, Vector> > &dat = *iter++;
    typename PointField::mesh_handle_type mag_mesh = detfld_->get_typed_mesh();
    

    PFieldNIter fld_iter, end;
    pair<PFieldNIter, PFieldNIter> iters = piters_[chunk++];
    fld_iter = iters.first;
    end = iters.second;

    int i = 0;
    // iterate over the detectors.
    while (fld_iter != end) {
      // set in field.
      PFieldNIndex ind = *fld_iter; 
      create_algo->set_val_scalar(magnitudes, ind, dat[i].first);
      fout->set_value(dat[i].second, ind);
      ++fld_iter; ++i;
    }
  }
}  

template <class ElecField, class CondField, class PointField>
void
CalcFMField<ElecField, CondField, PointField>::get_parallel_iter(int np)
{
  if (np_ != np) {
    np_ = np;
    interp_value_.clear();
    interp_value_.resize(np);
    typename PointField::mesh_handle_type mesh = detfld_->get_typed_mesh();

    PFieldNIter iter, end;
    mesh->begin(iter);
    mesh->end(end);
    //    cout << "begin end total: " << (unsigned)(*iter) << ", " 
    // << (unsigned)(*end) << endl;
    typename PointField::mesh_type::Node::size_type sz;
    mesh->size(sz);
    if (sz < (unsigned)np) {np_ = sz;}
    int chunk_sz = (int)ceil((float)sz / (float)np_);
    //cout << "chunk_size: " << chunk_sz 
    // << " mesh size: " << sz << endl;
    int i = 0;
    pair<PFieldNIter, PFieldNIter> iter_set;
    iter_set.first = iter;
    ++iter; ++i;
    while (iter != end) {
      if (i % chunk_sz == 0) {
	iter_set.second = iter;
	piters_.push_back(iter_set);
	iter_set.first = iter;
      }
      ++iter; ++i;
    }
    if (piters_.size() != (unsigned)np_) {
      ASSERT(piters_.size() < (unsigned)np_);
      iter_set.second = end;
      piters_.push_back(iter_set);
    }
    // make sure last chunk is end.
    piters_[np_-1].second = end;
  }
}

template <class ElecField, class CondField, class PointField>
void
CalcFMField<ElecField, CondField, PointField>::set_up_cell_cache()
{
  typename ElecField::mesh_handle_type mesh = efld_->get_typed_mesh();
  mesh->synchronize(Mesh::CELLS_E);

  EFieldCIter iter, end;
  mesh->begin(iter);
  mesh->end(end);

  while (iter != end) {
    typename ElecField::mesh_type::Cell::index_type cur_cell = *iter;
    ++iter;
    
    per_cell_cache c;
    mesh->get_center(c.center_, cur_cell);

    Vector elemField;
	
      // sanity check?
    efld_->value(elemField, cur_cell);
    int material = -1;
    ctfld_->value(material, cur_cell);

    c.cur_density_ = tens_[material].second * -1 * elemField; 
    c.volume_ = mesh->get_volume(cur_cell);
      
    cell_cache_[(unsigned)cur_cell] = c;
  }

}

template <class ElecField, class CondField, class PointField>
void
CalcFMField<ElecField, CondField, PointField>::interpolate(int proc, Point p)  
{

  typename ElecField::mesh_handle_type mesh = efld_->get_typed_mesh();
  mesh->synchronize(Mesh::CELLS_E);
  typename ElecField::mesh_type::Cell::index_type inside_cell;

  bool outside = ! mesh->locate(inside_cell, p);
  EFieldCIter iter, end;
  mesh->begin(iter);
  mesh->end(end);

  while (iter != end) {
    typename ElecField::mesh_type::Cell::index_type cur_cell = *iter;
    ++iter;
    
    if (outside || cur_cell != inside_cell) {

      per_cell_cache &c = (*cell_cache_.find((unsigned)cur_cell)).second;
      Vector radius = p - c.center_;
      
      Vector valueJXR = Cross(c.cur_density_, radius);
      double length = radius.length();
      

      interp_value_[proc] += ((valueJXR / (length * length * length)) * 
			      c.volume_); 
    }
  }
}

} // End namespace BioPSE


