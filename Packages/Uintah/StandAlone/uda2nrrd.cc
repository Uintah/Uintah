/*
 *  puda.cc: Print out a uintah data archive
 *
 *  Written by:
 *   James L. Bigler
 *   Department of Computer Science
 *   University of Utah
 *   April 2003
 *
 *  Copyright (C) 2003 U of U
 */

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/GenericField.h>


#include <Core/Math/MinMax.h>

#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/OS/Dir.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Persistent/Pstreams.h>

#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Dataflow/Modules/Selectors/PatchToField.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>

#include <teem/nrrd.h>

#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <algorithm>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
typedef LVMesh::handle_type LVMeshHandle;

bool verbose = false;
bool quiet = false;
bool attached_header = true;
bool remove_boundary = false;

enum {
  None,
  Det,
  Norm,
  Trace
};
int matrix_op = None;

class QueryInfo {
public:
  QueryInfo() {}
  QueryInfo(DataArchive* archive,
            LevelP level,
            string varname,
            int mat,
            double time,
            const Uintah::TypeDescription *type):
    archive(archive), level(level), varname(varname), mat(mat), time(time),
    type(type)
  {}
  
  DataArchive* archive;
  LevelP level;
  string varname;
  int mat;
  double time;
  const Uintah::TypeDescription *type;
};

void usage(const std::string& badarg, const std::string& progname)
{
  if(badarg != "")
    cerr << "Error parsing argument: " << badarg << endl;
  cerr << "Usage: " << progname << " [options] "
       << "-uda <archive file>\n\n";
  cerr << "Valid options are:\n";
  cerr << "  -h,--help  Prints this message out\n";
  
  cerr << "\nField Specifier Options\n";
  cerr << "  -v,--variable <variable name>\n";
  cerr << "  -m,--material <material number> [defaults to first material found]\n";
  cerr << "  -l,--level <level index> [defaults to 0]\n";
  cerr << "  -mo <operator> type of operator to apply to matricies.\n";
  cerr << "                 Options are none, det, norm, and trace\n";
  cerr << "                 [defaults to none]\n";
  cerr << "  -nbc,--noboundarycells - remove boundary cells from output\n";
  
  cerr << "\nOutput Options\n";
  cerr << "  -o,--out <outputfilename> [defaults to data]\n";
  cerr << "  -dh,--detatched-header - writes the data with detached headers.  The default is to not do this.\n";
  //    cerr << "  -binary (prints out the data in binary)\n";
  
  cerr << "\nTimestep Specifier Optoins\n";
  cerr << "  -tlow,--timesteplow [int] (only outputs timestep from int) [defaults to 0]\n";
  cerr << "  -thigh,--timestephigh [int] (only outputs timesteps up to int) [defaults to last timestep]\n";
  cerr << "  -tinc [int] (output every n timesteps) [defaults to 1]\n";
  cerr << "  -tstep,--timestep [int] (only outputs timestep int)\n";
  
  cerr << "\nChatty Options\n";
  cerr << "  -vv,--verbose (prints status of output)\n";
  cerr << "  -q,--quiet (very little output)\n";
  exit(1);
}


///////////////////////////////////////////////////////////////////
// Special nrrd functions
//
template <class T>
unsigned int get_nrrd_type();

template <>
unsigned int get_nrrd_type<char>() {
  return nrrdTypeChar;
}


template <>
unsigned int get_nrrd_type<unsigned char>()
{
  return nrrdTypeUChar;
}

template <>
unsigned int get_nrrd_type<short>()
{
  return nrrdTypeShort;
}

template <>
unsigned int get_nrrd_type<unsigned short>()
{
  return nrrdTypeUShort;
}

template <>
unsigned int get_nrrd_type<int>()
{
  return nrrdTypeInt;
}

template <>
unsigned int get_nrrd_type<unsigned int>()
{
  return nrrdTypeUInt;
}

template <>
unsigned int get_nrrd_type<long long>()
{
  return nrrdTypeLLong;
}

template <>
unsigned int get_nrrd_type<unsigned long long>()
{
  return nrrdTypeULLong;
}

template <>
unsigned int get_nrrd_type<float>()
{
  return nrrdTypeFloat;
}

template <class T>
unsigned int get_nrrd_type() {
  return nrrdTypeDouble;
}

/////////////////////////////////////////////////////////////////////
//



//
/////////////////////////////////////////////////////////////////////



/////////////////////////////////////////////////////////////////////
//
// This will gather all the patch data and write it into the single
// contiguous volume.
template <class T, class VarT, class FIELD>
void build_field(QueryInfo &qinfo,
                 IntVector& offset,
                 T& /* data_type */,
                 VarT& /*var*/,
                 FIELD *sfield)
{
  // TODO: Bigler
  // Not sure I need this yet
  sfield->fdata().initialize(T(0));

  // Loop over each patch and get the data from the data archive.
  for( Level::const_patchIterator patch_it = qinfo.level->patchesBegin();
       patch_it != qinfo.level->patchesEnd(); ++patch_it){
    IntVector patch_low, patch_high;
    VarT patch_data;
    const Patch* patch = *patch_it;
    // This gets the data
    qinfo.archive->query( patch_data, qinfo.varname, qinfo.mat, patch,
                          qinfo.time);
    if ( remove_boundary ) {
      if(sfield->basis_order() == 0){
        patch_low = patch->getInteriorCellLowIndex();
        patch_high = patch->getInteriorCellHighIndex();
      } else {
        patch_low = patch->getInteriorNodeLowIndex();
        switch (qinfo.type->getType()) {
        case Uintah::TypeDescription::SFCXVariable:
          patch_high = patch->getInteriorHighIndex(Patch::XFaceBased);
          break;
        case Uintah::TypeDescription::SFCYVariable:
          patch_high = patch->getInteriorHighIndex(Patch::YFaceBased);
          break;
        case Uintah::TypeDescription::SFCZVariable:
          patch_high = patch->getInteriorHighIndex(Patch::ZFaceBased);
          break;
        default:
          patch_high = patch->getInteriorNodeHighIndex();   
        } 
      }
      
    } else { // Don't remove the boundary
      if( sfield->basis_order() == 0){
        patch_low = patch->getCellLowIndex();
        patch_high = patch->getCellHighIndex();
      } else {
        patch_low = patch->getNodeLowIndex();
        switch (qinfo.type->getType()) {
        case Uintah::TypeDescription::SFCXVariable:
          patch_high = patch->getSFCXHighIndex();
          break;
        case Uintah::TypeDescription::SFCYVariable:
          patch_high = patch->getSFCYHighIndex();
          break;
        case Uintah::TypeDescription::SFCZVariable:
          patch_high = patch->getSFCZHighIndex();
          break;
        case Uintah::TypeDescription::NCVariable:
          patch_high = patch->getNodeHighIndex();
          break;
        default:
          cerr << "build_field::unknown variable.\n";
          exit(1);
        }
      }
    } // if (remove_boundary)

    // Rewindow the data if we need only a subset.  This should never
    // get bigger (thus requiring reallocation).  rewindow returns
    // true iff no reallocation is needed.
    ASSERT(patch_data.rewindow( patch_low, patch_high ));
    
    PatchToFieldThread<VarT, T, FIELD> *worker = 
      scinew PatchToFieldThread<VarT, T, FIELD>(sfield, patch_data, offset,
                                                patch_low, patch_high);
    worker->run();
    delete worker;
  }
}

// helper function for wrap_nrrd

template <class T>
bool 
wrap_copy( T* fdata, double*& datap, unsigned int size){
  cerr<<"Should not be called for scalar data, no copy required!";
  return false;
}

// Vector version
template <>
bool
wrap_copy( Vector* fdata, double*& datap, unsigned int size){

  // Copy the data
  for(unsigned int i = 0; i < size; i++) {
    *datap++ = fdata->x();
    *datap++ = fdata->y();
    *datap++ = fdata->z();
    fdata++;
  }
  return true;
}

// Matrix3 version

template <>
bool
wrap_copy( Matrix3* fdata, double*& datap, unsigned int size){

  switch (matrix_op) {
  case None:
    for(unsigned int i = 0; i < size; i++) {
      for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
          *datap++ = (*fdata)(i,j);
      fdata++;
    }
    break;
  case Det:
    for(unsigned int i = 0; i < size; i++) {
      *datap++ = fdata->Determinant();
      fdata++;
    }
    break;
  case Trace:
    for(unsigned int i = 0; i < size; i++) {
      *datap++ = fdata->Trace();
      fdata++;
    }
    break;
  case Norm:
    for(unsigned int i = 0; i < size; i++) {
      *datap++ = fdata->Norm();
      fdata++;
    }
    break;
  default:
    cerr << "Unknown matrix operation\n";
    return false;
  }
  return true;
}

// Allocates memory for dest, then copies all the data to dest from
// source.
template<class FIELD>
Nrrd* wrap_nrrd(FIELD *source) {
  Nrrd *out = nrrdNew();
  int dim = 3;
  size_t size[5];
  
  size[0] = source->fdata().dim3();
  size[1] = source->fdata().dim2();
  size[2] = source->fdata().dim1();

  if (verbose) for(int i = 0; i < dim; i++) cout << "size["<<i<<"] = "<<size[i]<<endl;

  const SCIRun::TypeDescription *td = 
    source->get_type_description(Field::FDATA_TD_E);
  if( td->get_name().find( "Vector") != string::npos ) {  // Vectors
    unsigned int num_vec = source->fdata().size();
    double *data = new double[num_vec*3];
    if (!data) {
      cerr << "Cannot allocate memory ("<<num_vec*3*sizeof(double)<<" byptes) for temp storage of vectors\n";
      nrrdNix(out);
      return 0;
    }
    double *datap = data;
    typename FIELD::value_type *vec_data = &(source->fdata()(0,0,0));
    

    // Copy the data
    wrap_copy( vec_data, datap, num_vec);
    
    if (nrrdWrap_nva(out, data, nrrdTypeDouble, dim, size) == 0) {
      return out;
    } else {
      nrrdNix(out);
      delete data;
      return 0;
    }
  } else if (td->get_name().find( "Matrix3") != string::npos ) { // Matrix3
    dim = matrix_op == None? 5 : 3;
    if (matrix_op == None) {
      size[0] = 3;
      size[1] = 3;
      size[2] = source->fdata().dim3();
      size[3] = source->fdata().dim2();
      size[4] = source->fdata().dim1();
    } else {
      size[0] = source->fdata().dim3();
      size[1] = source->fdata().dim2();
      size[2] = source->fdata().dim1();
    }
    unsigned int num_mat = source->fdata().size();
    int elem_size = matrix_op == None? 9 : 1;
    double *data = new double[num_mat*elem_size];
    if (!data) {
      cerr << "Cannot allocate memory ("<<num_mat*elem_size*sizeof(double)<<" byptes) for temp storage of vectors\n";
      nrrdNix(out);
      return 0;
    }
    double *datap = data;
    typename FIELD::value_type *mat_data = &(source->fdata()(0,0,0));
    // Copy the data
    if( !wrap_copy( mat_data, datap, num_mat) ){
      nrrdNix(out);
      delete data;
      return 0;
    }

    if (nrrdWrap_nva(out, data, nrrdTypeDouble, dim, size) == 0) {
      return out;
    } else {
      nrrdNix(out);
      delete data;
      return 0;
    }
  } else { // Scalars
    // We don't need to copy data, so just get the pointer to the data
    size_t field_size = (source->fdata().size() *
                         sizeof(typename FIELD::value_type));
    void* data = malloc(field_size);
    if (!data) {
      cerr << "Cannot allocate memory ("<<field_size<<" byptes) for scalar nrrd copy.\n";
      nrrdNix(out);
      return 0;
    }
    memcpy(data, (void*)&(source->fdata()(0,0,0)), field_size);

    if (nrrdWrap_nva(out, data, get_nrrd_type< typename FIELD::value_type>(), 
                     dim, size) == 0) {
      return out;
    } else {
      nrrdNix(out);
      free(data);
      return 0;
    }
  }
}


// getData<CCVariable<T>, T >();
template<class VarT, class T>
void getData(QueryInfo &qinfo, IntVector &low,
             LVMeshHandle mesh_handle_,
             int basis_order,
             string &filename) {

  typedef GenericField<LVMesh, ConstantBasis<T>,
                       FData3d<T, LVMesh> > LVFieldCB;
  typedef GenericField<LVMesh, HexTrilinearLgn<T>, 
                       FData3d<T, LVMesh> > LVFieldLB;

  VarT gridVar;
  T data_T;

  // set the generation and timestep in the field
  if (!quiet) cout << "Building Field from uda data\n";
  
  // Print out the psycal extents
  BBox bbox = mesh_handle_->get_bounding_box();
  if (!quiet) cout << "Bounding box: min("<<bbox.min()<<"), max("<<bbox.max()<<")\n";
  

  // Get the nrrd data, and print it out.
  char *err;

  Nrrd *out;
  if( basis_order == 0 ){
    LVFieldCB* sf = scinew LVFieldCB(mesh_handle_);
    if (!sf) {
      cerr << "Cannot allocate memory for field\n";
      return;
    }
    build_field(qinfo, low, data_T, gridVar, sf);
    // Convert the field to a nrrd
    out = wrap_nrrd(sf);
    // Clean up our memory
    delete sf;
  } else {
    LVFieldLB* sf = scinew LVFieldLB(mesh_handle_);
    if (!sf) {
      cerr << "Cannot allocate memory for field\n";
      return;
    }
    build_field(qinfo, low, data_T, gridVar, sf);
    // Convert the field to a nrrd
    out = wrap_nrrd(sf);
    // Clean up our memory
    delete sf;
  }

#if 0
  Piostream *fieldstrm =
    scinew BinaryPiostream(string(filename + ".fld").c_str(),
                           Piostream::Write);
  if (fieldstrm->error()) {
    cerr << "Could not open test.fld for writing.\n";
    exit(1);
  } else {
    Pio(*fieldstrm, *source_field);
    delete fieldstrm;
  }
#endif

  if (out) {
    // Now write it out
    if (!quiet) cout << "Writing nrrd file\n";
    string filetype = attached_header? ".nrrd": ".nhdr";
    if (nrrdSave(string(filename + filetype).c_str(), out, 0)) {
      // There was a problem
      err = biffGetDone(NRRD);
      cerr << "Error writing nrrd:\n"<<err<<"\n";
    } else {
      if (!quiet) cout << "Done writing nrrd file\n";
    }
    // nrrdNuke deletes the nrrd and the data inside the nrrd
    nrrdNuke(out);
  } else {
    // There was a problem
    err = biffGetDone(NRRD);
    cerr << "Error wrapping nrrd: "<<err<<"\n";
  }  
  return;
}

// getVariable<double>();
template<class T>
void getVariable(QueryInfo &qinfo, IntVector &low, IntVector& hi,
                 IntVector &range, BBox &box,
                 string &filename) {

  LVMeshHandle mesh_handle_;
  switch( qinfo.type->getType() ) {
  case Uintah::TypeDescription::CCVariable:
    mesh_handle_ = scinew LVMesh(range.x(), range.y(),
                                 range.z(), box.min(),
                                 box.max());
    getData<CCVariable<T>, T>(qinfo, low, mesh_handle_, 0,
                              filename);
    break;
  case Uintah::TypeDescription::NCVariable:
    mesh_handle_ = scinew LVMesh(range.x(), range.y(),
                                 range.z(), box.min(),
                                 box.max());
    getData<NCVariable<T>, T>(qinfo, low, mesh_handle_, 1,
                              filename);
    break;
  case Uintah::TypeDescription::SFCXVariable:
    mesh_handle_ = scinew LVMesh(range.x(), range.y()-1,
                                 range.z()-1, box.min(),
                                 box.max());
    getData<SFCXVariable<T>, T>(qinfo, low, mesh_handle_, 1,
                                filename);
    break;
  case Uintah::TypeDescription::SFCYVariable:
    mesh_handle_ = scinew LVMesh(range.x()-1, range.y(),
                                 range.z()-1, box.min(),
                                 box.max());
    getData<SFCYVariable<T>, T>(qinfo, low, mesh_handle_, 1,
                                filename);
    break;
  case Uintah::TypeDescription::SFCZVariable:
    mesh_handle_ = scinew LVMesh(range.x()-1, range.y()-1,
                                 range.z(), box.min(),
                                 box.max());
    getData<SFCZVariable<T>, T>(qinfo, low, mesh_handle_, 1,
                                filename);
    break;
  default:
    cerr << "Type is unknown.\n";
    return;
    break;
  
  }
}

bool
is_periodic_bcs(IntVector cellir, IntVector ir)
{
  if( cellir.x() == ir.x() ||
      cellir.y() == ir.y() ||
      cellir.z() == ir.z() )
    return true;
  else
    return false;
}

void
get_periodic_bcs_range(IntVector cellmax, IntVector datamax,
                       IntVector range, IntVector& newrange)
{
  if( cellmax.x() == datamax.x())
    newrange.x( range.x() + 1 );
  else
    newrange.x( range.x() );
  if( cellmax.y() == datamax.y())
    newrange.y( range.y() + 1 );
  else
    newrange.y( range.y() );
  if( cellmax.z() == datamax.z())
    newrange.z( range.z() + 1 );
  else
    newrange.z( range.z() );
}

int main(int argc, char** argv)
{
  /*
   * Default values
   */
  bool do_binary=false;

  unsigned long time_step_lower = 0;
  // default to be last timestep, but can be set to 0
  unsigned long time_step_upper = (unsigned long)-1;
  unsigned long tinc = 1;

  string input_uda_name;
  string output_file_name("");
  bool use_default_file_name = true;
  IntVector var_id(0,0,0);
  string variable_name("");
  // It will use the first material found unless other indicated.
  int material = -1;
  int level_index = 0;
  
  /*
   * Parse arguments
   */
  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-l" || s == "--level") {
      level_index = atoi(argv[++i]);
    } else if (s == "-vv" || s == "--verbose") {
      verbose = true;
    } else if (s == "-q" || s == "--quiet") {
      quiet = true;
    } else if (s == "-tlow" || s == "--timesteplow") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-thigh" || s == "--timestephigh") {
      time_step_upper = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-tstep" || s == "--timestep") {
      time_step_lower = strtoul(argv[++i],(char**)NULL,10);
      time_step_upper = time_step_lower;
    } else if (s == "-tinc") {
      tinc = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-i" || s == "--index") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_id = IntVector(x,y,z);
    } else if( s ==  "-dh" || s == "--detatched-header") {
      attached_header = false;
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
      use_default_file_name = false;
    } else if(s == "-mo") {
      s = argv[++i];
      if (s == "det")
        matrix_op = Det;
      else if (s == "norm")
        matrix_op = Norm;
      else if (s == "trace")
        matrix_op = Trace;
      else if (s == "none")
        matrix_op = None;
      else
        usage(s, argv[0]);
    } else if(s == "-binary") {
      do_binary=true;
    } else if(s == "-nbc" || s == "--noboundarycells") {
      remove_boundary = true;
    } else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    DataArchive* archive = scinew DataArchive(input_uda_name);

    ////////////////////////////////////////////////////////
    // Get the times and indices.

    vector<int> index;
    vector<double> times;
    
    // query time info from dataarchive
    archive->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    if (!quiet) cout << "There are " << index.size() << " timesteps:\n";
    
    //////////////////////////////////////////////////////////
    // Get the variables and types
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    if (verbose) cout << "There are " << vars.size() << " variables:\n";
    bool var_found = false;
    unsigned int var_index = 0;
    for (;var_index < vars.size(); var_index++) {
      if (variable_name == vars[var_index]) {
        var_found = true;
        break;
      }
    }
    
    if (!var_found) {
      cerr << "Variable \"" << variable_name << "\" was not found.\n";
      cerr << "If a variable name was not specified try -v [name].\n";
      cerr << "Possible variable names are:\n";
      var_index = 0;
      for (;var_index < vars.size(); var_index++) {
        cout << "vars[" << var_index << "] = " << vars[var_index] << endl;
      }
      cerr << "Aborting!!\n";
      exit(-1);
      //      var = vars[0];
    }

    if (use_default_file_name) {
      // Then use the variable name for the output name
      output_file_name = variable_name;
      if (!quiet)
        cout << "Using variable name ("<<output_file_name
             << ") as output file base name.\n";
    }
    
    if (!quiet) cout << "Extracing data for "<<vars[var_index] << ": " << types[var_index]->getName() <<endl;

    /////////////////////////////////////////////////////
    // figure out the lower and upper bounds on the timesteps
    if (time_step_lower >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      exit(1);
    }
    
    // set default max time value
    if (time_step_upper == (unsigned long)-1) {
      if (verbose)
        cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
      time_step_upper = times.size() - 1;
    }
    
    if (time_step_upper >= times.size() || time_step_upper < time_step_lower) {
      cerr << "timestephigh("<<time_step_lower<<") must be greater than " << time_step_lower 
           << " and less than " << times.size()-1 << endl;
      exit(1);
    }
    
    if (!quiet) cout << "outputting for times["<<time_step_lower<<"] = " << times[time_step_lower]<<" to times["<<time_step_upper<<"] = "<<times[time_step_upper] << endl;

    ////////////////////////////////////////////////////////
    // Loop over each timestep
    for (unsigned long time = time_step_lower; time <= time_step_upper;
         time+=tinc){

      // Check the level index
      double current_time = times[time];
      GridP grid = archive->queryGrid(current_time);
      if (level_index >= grid->numLevels() || level_index < 0) {
        cerr << "level index is bad ("<<level_index<<").  Should be between 0 and "<<grid->numLevels()<<".\n";
        cerr << "Trying next timestep.\n";
        continue;
      }
    
      ///////////////////////////////////////////////////
      // Check the material number.

      LevelP level = grid->getLevel(level_index);
      const Patch* patch = *(level->patchesBegin());
      ConsecutiveRangeSet matls =
        archive->queryMaterials(variable_name, patch, current_time);

      if (verbose) {
        // Print out all the material indicies valid for this timestep
        cout << "Valid materials for "<<variable_name<<" at time["<<time<<"]("<<current_time<<") are \n\t";
        for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
             matlIter != matls.end(); matlIter++) {
          cout << *matlIter << ", ";
        }
        cout << endl;
      }
      
      int mat_num;
      if (material == -1) {
        mat_num = *(matls.begin());
      } else {
        unsigned int mat_index = 0;
        mat_num = 0;
        for (ConsecutiveRangeSet::iterator matlIter = matls.begin();
             matlIter != matls.end(); matlIter++){
          int matl = *matlIter;
          if (matl == material) {
            mat_num = matl;
            break;
          }
          mat_index++;
        }
        if (mat_index == matls.size()) {
          // then we didn't find the right material
          cerr << "Didn't find material " << material << " in the data.\n";
          cerr << "Trying next timestep.\n";
          continue;
        }
      }
      if (!quiet) cout << "Extracting data for material "<<mat_num<<".\n";
      
      // get type and subtype of data
      const Uintah::TypeDescription* td = types[var_index];
      const Uintah::TypeDescription* subtype = td->getSubType();
    
      QueryInfo qinfo(archive, level, variable_name, mat_num, current_time,
                      td);

      IntVector hi, low, range;
      BBox box;

      // Remove the edges if no boundary cells
      if( remove_boundary ){
        level->findInteriorIndexRange(low, hi);
        level->getInteriorSpatialRange(box);
      } else {
        level->findIndexRange(low, hi);
        level->getSpatialRange(box);
      }
      range = hi - low;

      if (qinfo.type->getType() == Uintah::TypeDescription::CCVariable) {
        IntVector cellLo, cellHi;
        if (remove_boundary) {
          level->findInteriorCellIndexRange(cellLo, cellHi);
        } else {
          level->findCellIndexRange(cellLo, cellHi);
        }
        if (is_periodic_bcs(cellHi, hi)) {
          IntVector newrange(0,0,0);
          get_periodic_bcs_range( cellHi, hi, range, newrange);
          range = newrange;
        }
      }
      
      
      // Figure out the filename
      char filename_num[200];
      if (use_default_file_name)
        sprintf(filename_num, "_M%02d_%04lu", mat_num, time);
      else
        sprintf(filename_num, "_%04lu", time);
      string filename(output_file_name + filename_num);
    
      switch (subtype->getType()) {
      case Uintah::TypeDescription::double_type:
        getVariable<double>(qinfo, low, hi, range, box,
                            filename);
        break;
      case Uintah::TypeDescription::float_type:
        getVariable<float>(qinfo, low, hi, range, box,
                           filename);
        break;
      case Uintah::TypeDescription::int_type:
        getVariable<int>(qinfo, low, hi, range, box,
                         filename);
        break;
      case Uintah::TypeDescription::Vector:
        getVariable<Vector>(qinfo, low, hi, range, box,
                            filename);
        break;
      case Uintah::TypeDescription::Matrix3:
        getVariable<Matrix3>(qinfo, low, hi, range, box,
                             filename);
        break;
      case Uintah::TypeDescription::bool_type:
      case Uintah::TypeDescription::short_int_type:
      case Uintah::TypeDescription::long_type:
      case Uintah::TypeDescription::long64_type:
        cerr << "Subtype "<<subtype->getName()<<" is not implemented\n";
        exit(1);
        break;
      default:
        cerr << "Unknown subtype\n";
        exit(1);
      }
    } // end time step iteration
    
  } catch (Exception& e) {
    cerr << "Caught exception: " << e.message() << endl;
    exit(1);
  } catch(...){
    cerr << "Caught unknown exception\n";
    exit(1);
  }
}
