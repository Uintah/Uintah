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
    np_(-1),
    interp_value_(0)
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
  typedef typename PointField::mesh_type::Node::iterator PFieldNIter;
  typedef typename PointField::mesh_type::Node::index_type PFieldNIndex;

  void interpolate(int proc, Point p, Vector value, int num_threads);

  pair<EFieldCIter, EFieldCIter> get_parallel_iter(ElecField *fld, 
						   int proc, int np);

  int                                                     np_;
  Vector                                                  interp_value_;
  map<typename ElecField::mesh_type::Cell::index_type, Vector> current_density_;
  ElecField                                               *efld_;
  CondField                                               *ctfld_;
  vector<pair<string, Tensor> >                            tens_;
  vector<pair<EFieldCIter, EFieldCIter> >                  piters_;
};

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
  // interpolate needs these.
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
  PointField *detfld = dynamic_cast<PointField*>(detectors.get_rep());
  ASSERT(detfld != 0);
  
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

  PFieldNIter iter, end;
  mesh->begin(iter);
  mesh->end(end);
  
  // iterate over the detectors.
  while (iter != end) {
    // finish loop iteration.

    PFieldNIndex ind = *iter;
    ++iter;

    Vector mag_field;
    Point  pt;
    mesh->get_center(pt, ind);
    
    // switch to parallel2 mag_field arg unused in interp FIX_ME
    cerr << "Number of Processors Used: " << np <<endl;
    
    // init the interp val to 0 
    interp_value_ = Vector(0,0,0);
    Thread::parallel(Parallel3<AlgoType, Point, Vector, int>(this, 
						     &AlgoType::interpolate, 
						     pt, mag_field, np), 
		     np, true);
    mag_field = interp_value_;

    //interpolate(efield, pt, mag_field, sigma, np);

    Vector normal;
    normal = detfld->value(ind);
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
      Vector P   = dipfld->value(dip_ind);
      
      Vector radius = pt - pt2; // detector - source

      Vector valuePXR = Cross(P, radius);
      double length = radius.length();
 
      mag_field += valuePXR / (length * length * length);
      
    }
    // end of B(P) stuff
    const double one_over_4_pi = 1.0 / (4 * M_PI);
    mag_field *= one_over_4_pi;

    // set in field.
    fout->set_value(mag_field, ind);
    
    //use Dot for simulations & length for testing with sphere
    create_algo->set_val_scalar(magnitudes, ind, Dot(mag_field, normal));
  }
  return true;
}

inline
Vector 
mag_mult(vector<double> matrix, Vector elemField) {
  return(Vector(
     matrix[0]*elemField.x()+matrix[1]*elemField.y()+matrix[2]*elemField.z(),
     matrix[1]*elemField.x()+matrix[3]*elemField.y()+matrix[4]*elemField.z(),
     matrix[2]*elemField.x()+matrix[4]*elemField.y()+matrix[5]*elemField.z()));
}


template <class ElecField, class CondField, class PointField>
pair<typename CalcFMField<ElecField, CondField, PointField>::EFieldCIter, 
     typename CalcFMField<ElecField, CondField, PointField>::EFieldCIter>
CalcFMField<ElecField, CondField, PointField>::get_parallel_iter(
							      ElecField *fld,
							      int proc, 
							      int np)
{
  if (np_ != np) {
    np_ = np;
    typename ElecField::mesh_handle_type mesh = fld->get_typed_mesh();

    EFieldCIter iter, end;
    mesh->begin(iter);
    mesh->end(end);
    cout << "begin end total: " << (unsigned)(*iter) << ", " 
	     << (unsigned)(*end) << endl;
    typename ElecField::mesh_type::Cell::size_type sz;
    mesh->size(sz);
    int chunk_sz = sz / np;
    cout << "chunk_size: " << chunk_sz 
	 << " mesh size: " << sz << endl;
    int i = 0;
    pair<EFieldCIter, EFieldCIter> iter_set;
    iter_set.first = iter;
    ++iter; ++i;
    while (iter != end) {
      if (i % chunk_sz == 0) {
	piters_.push_back(iter_set);
	iter_set.first = iter;
	++iter; ++i;
      }
      if (iter == end) { break; }
      iter_set.second = iter;
      ++iter; ++i;
    }
    if (piters_.size() != (unsigned)np) {
      ASSERT(piters_.size() < (unsigned)np);
      iter_set.second = end;
      piters_.push_back(iter_set);
    }
    // make sure last chunk is end.
    piters_[np-1].second = end;
  }
  return piters_[proc];
}  

template <class ElecField, class CondField, class PointField>
void
CalcFMField<ElecField, CondField, PointField>::interpolate(int proc, 
							   Point p, 
							   Vector value, 
							   int np)
{

  typename ElecField::mesh_handle_type mesh = efld_->get_typed_mesh();
  mesh->synchronize(Mesh::CELLS_E);
  typename ElecField::mesh_type::Cell::index_type inside_cell;
  
  bool outside = ! mesh->locate(inside_cell, p);
  EFieldCIter iter;
  pair<EFieldCIter, EFieldCIter> iters = get_parallel_iter(efld_, proc, np);
  cout << "interp begin end pair: " << (unsigned)(*iters.first) << ", " 
       << (unsigned)(*iters.second) << endl;

  iter = iters.first;
  while (iter != iters.second) {
    typename ElecField::mesh_type::Cell::index_type cur_cell = *iter;
    ++iter;
    
    if (outside || cur_cell != inside_cell) {

      Point centroid;
      mesh->get_center(centroid, cur_cell);

      if (current_density_[cur_cell] != Vector()) {
	Vector elemField;

	// sanity check?
	efld_->value(elemField, cur_cell);
	int material = -1;
	ctfld_->value(material, cur_cell);
	Vector condElect = tens_[material].second * -1 * elemField; 

	//lock
	current_density_[cur_cell] = condElect;
	//unlock
      }
    
		  
      Vector radius = p - centroid;
      
      Vector valueJXR = Cross(current_density_[cur_cell], radius);
      double length = radius.length();
      
      Vector tmp = ((valueJXR / (length * length * length)) * 
		    mesh->get_volume(cur_cell));
      //lock
      interp_value_ += tmp;
      //unlock
    }
  }
}

} // End namespace BioPSE


