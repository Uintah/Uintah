/*
 *  uintahparticle.cc: Print out a uintah data archive
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   June 2000
 *
 *  Copyright (C) 2000 U of U
 */

// rtrt libraries
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/CatmullRomSpline.h>
#include <Packages/rtrt/Core/GridSpheres.h>
#include <Packages/rtrt/Core/GridSpheresDpy.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/LambertianMaterial.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/TimeObj.h>
#include <Packages/rtrt/Core/Array1.cc>
#if 0
#undef Exception
#undef SCI_ASSERTION_LEVEL
#undef ASSERTEQ
#undef ASSERTRANGE
#undef ASSERTL1
#undef ASSERTL2
#undef ASSERTL3
#endif
#undef None

#ifdef Success
#undef Success
#endif

//CSAFE libraries
#include <Core/Math/MinMax.h>
#include <Core/Exceptions/Exception.h>
//using SCICore::Exceptions::Exception;
using namespace SCIRun;
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/CCA/Ports/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/ShareAssignParticleVariable.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Dataflow/XMLUtil/XMLUtil.h>
// general
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace Uintah;
using namespace rtrt;

bool debug = false;

// don't know if this is the way to do it --bigler
extern "C" void AuditDefaultAllocator() {
}

extern "C" void audit() {
  AuditDefaultAllocator();
}

class SphereData {
public:
  float* data;
  int nspheres;
  int numvars;
  float radius;
  rtrt::Array1<float> mins;
  rtrt::Array1<float> maxs;
};

struct MaterialData {
  int material_index;
  vector<ShareAssignParticleVariable<double> > pv_double_list;
  vector<ShareAssignParticleVariable<int> > pv_int_list;
  vector<ShareAssignParticleVariable<SCIRun::Point> > pv_point_list;
  vector<ShareAssignParticleVariable<SCIRun::Vector> > pv_vector_list;
  vector<ShareAssignParticleVariable<SCIRun::Matrix3> > pv_matrix_list;
  ShareAssignParticleVariable<SCIRun::Point> p_x;
  MaterialData();
  ~MaterialData();
};

MaterialData::MaterialData(): material_index(0) {}

MaterialData::~MaterialData()
{
  #if 0
  cerr << "Destroying: " << this << '\n';
  AuditDefaultAllocator();
  cerr << "Passed audit\n";
  #endif
}

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
	cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options] <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -PTvar\n";
    cerr << "  -ptonly (outputs only the point locations\n";
    cerr << "  -patch (outputs patch id with data)\n";
    cerr << "  -material (outputs material number with data)\n";
    cerr << "  -verbose (prints status of output)\n";
    cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
    cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
    cerr << "  -gridcellsize [int]\n";
    cerr << "  -griddepth [int]\n";
    cerr << "  -colordata [int]\n";
    cerr << "  -radiusfactor [float]\n";
    cerr << "  -radius [float]\n";
    cerr << "  -rate [float]\n";
    cerr << "*NOTE* ptonly, patch, material, timesteplow, timestephigh \
are used in conjuntion with -PTvar.\n";
    
    return;
}

void get_material(rtrt::Array1<Material*> &matls) {
  CatmullRomSpline<Color> spline(0);
  spline.add(Color(.4,.4,.4));
  spline.add(Color(.4,.4,1));
  //    for(int i=0;i<2;i++)
  spline.add(Color(.4,1,.4));
  //    for(int i=0;i<3;i++)
  spline.add(Color(1,1,.4));
  //    for(int i=0;i<300;i++)
  spline.add(Color(1,.4,.4));
  int ncolors=5000;
  matls.resize(ncolors);
  float Ka=.8;
  float Kd=.8;
  float Ks=.8;
  float refl=0;
  float specpow=40;
  for(int i=0;i<ncolors;i++){
    float frac=float(i)/(ncolors-1);
    Color c(spline(frac));
    matls[i]=new Phong(c*Ka, c*Kd, c*Ks, specpow, refl);
    //matls[i]=new LambertianMaterial(c*Kd);
  }
}

#ifdef PATCH_BY_PATCH
void processdata(vector<MaterialData> material_data_list,
		 const Uintah::Patch* patch,
		 bool do_PTvar_all, bool do_patch, bool do_material,
		 bool do_verbose, int colordata, float radius,
		 float radius_factor, GridSpheresDpy* display,
		 Group* timeblock,
		 rtrt::Array1<rtrt::Material*> material_properties,
		 int gridcellsize, int griddepth) {
  AuditDefaultAllocator();
  if (debug) cerr << "Start of processdata\n";
  //--------------------------------------------------
  // set up the first min/max
  // and determine number of particles and variables
  SCIRun::Point min, max;
  vector<double> d_min,d_max,v_min,v_max;
  bool data_found = false;
  long num_particles = 0;
  int num_variables = 0;
  
  // loops until a non empty material_data set has been
  // found and inialized the mins and maxes
  for(int m = 0; m < material_data_list.size(); m++) {
    MaterialData md = material_data_list[m];
    ParticleSubset* pset = md.p_x.getParticleSubset();
    if (!pset) {
      cerr << "No particle location variable found\n";
      abort();
    }
    int numParticles = pset->numParticles();
    if(numParticles > 0){
      num_particles+=numParticles;
      ParticleSubset::iterator iter = pset->begin();
      
      if (!data_found) {
	// setup for p.x
	min=max=md.p_x[*iter];
	num_variables += 3;
	// setup for all others
	if (do_PTvar_all) {
	  for(int i = 0; i < md.pv_double_list.size(); i++) {
	    d_min.push_back(md.pv_double_list[i][*iter]);
	    d_max.push_back(md.pv_double_list[i][*iter]);
	    num_variables++;
	  }
	  for(int i = 0; i < md.pv_vector_list.size(); i++) {
	    v_min.push_back(md.pv_vector_list[i][*iter].length());
	    v_max.push_back(md.pv_vector_list[i][*iter].length());
	    num_variables++;
	  }
	}
	// initialized mins/maxes
	data_found = true;
      }
    } // end if(numParticles > 0)
  } // end material_data_list loop
  
  if (do_patch)
    num_variables++;
  if (do_material)
    num_variables++;
  if (do_verbose) {
    cerr << "Number of variables found: " << num_variables << endl;
    cerr << "Number of particles found in patch: " << num_particles << endl;
  }
  //--------------------------------------------------
  // extract data and write it to a file MaterialData at a time
  
  //---------
  // allocate memory for particle data
  float* data = (float*)malloc(num_variables*num_particles*sizeof(float));
  //	    if (do_verbose)
  //cerr << "---Extracting data and writing it out  ";
  float* p=data;
  if (do_verbose) cerr << "Past data allocation\n";
  for(int m = 0; m < material_data_list.size(); m++) {
    MaterialData md = material_data_list[m];
    ParticleSubset* pset = md.p_x.getParticleSubset();
    // a little redundant, but may not have been cought
    // by the previous section
    if (!pset) {
      cerr << "No particle location variable found\n";
      abort();
    }
    
    int numParticles = pset->numParticles();
    if (do_verbose) cerr << "m = " << m << "  numParticles = " << numParticles << endl;
    //total_particles+= numParticles;
    if(numParticles > 0){
      ParticleSubset::iterator iter = pset->begin();
      for(;iter != pset->end(); iter++){
	// p_x
	min=Min(min, md.p_x[*iter]);
	max=Max(max, md.p_x[*iter]);
	float temp_value = (float)(md.p_x[*iter]).x();
	*p++=temp_value;
	//fwrite(&temp_value, sizeof(float), 1, datafile);
	temp_value = (float)(md.p_x[*iter]).y();
	*p++=temp_value;
	
	//fwrite(&temp_value, sizeof(float), 1, datafile);
	temp_value = (float)(md.p_x[*iter]).z();
	*p++=temp_value;
	//fwrite(&temp_value, sizeof(float), 1, datafile);
	if (do_PTvar_all) {
	  // double data
	  for(int i = 0; i < md.pv_double_list.size(); i++) {
	    double value = md.pv_double_list[i][*iter];
	    d_min[i]=SCIRun::Min(d_min[i],value);
	    d_max[i]=SCIRun::Max(d_max[i],value);
	    temp_value = (float)value;
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  // vector data
	  for(int i = 0; i < md.pv_vector_list.size(); i++) {
	    double value = md.pv_vector_list[i][*iter].length();
	    v_min[i]=SCIRun::Min(v_min[i],value);
	    v_max[i]=SCIRun::Max(v_max[i],value);
	    temp_value = (float)value;
	    *p++=temp_value;
	    
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  if (do_patch) {
	    temp_value = (float)patch->getID();
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  if (do_material) {
	    temp_value = (float)m;
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	}
      }
    }
  }
  
  AuditDefaultAllocator();
  if (debug) cerr << "Past data write, on to datasummary.\n";
  //--------------------------------------------------
  // setup data summary and create GridSphere
  float* mins = (float*)malloc(num_variables*sizeof(float));
  float* maxs = (float*)malloc(num_variables*sizeof(float));
  int v = 0;
  if (debug) cerr << "data_found = " << data_found << "\n";
  if (data_found) {
    mins[0] = min.x();
    mins[1] = min.y();
    mins[2] = min.z();
    maxs[0] = max.x();
    maxs[1] = max.y();
    maxs[2] = max.z();
    v = 3;
    if (do_PTvar_all) {
      for(int i = 0; i < d_min.size(); i++) {
	mins[v] = d_min[i];
	maxs[v] = d_max[i];
	v++;
      }
      for(int i = 0; i < v_min.size(); i++) {
	mins[v] = v_min[i];
	maxs[v] = v_max[i];
	v++;
      }
      if (do_patch) {
	mins[v] = (float)patch->getID();
	maxs[v] = (float)patch->getID();
	v++;
      }
      if (do_material) {
	mins[v] = 0;
	maxs[v] = (float)material_data_list.size() ;
      }
    }
#if 0 // this is now done in GridSpheresDpy
    // this takes into account of
    // min/max equaling each other
    for(int i=0; i < num_variables; i++) {
      if (mins[i] == maxs[i]) {
	if (maxs[i] > 0) {
	  maxs[i]*=1.1;
	} else {
	  if (maxs[i] < 0)
	    maxs[i]*=0.9;
	  else
	    maxs[i]=1;
	}
      }
    }
#endif
    // since colordata is used to index into arrays the size
    // of num_variables, it must be bounds checked.
    if(colordata<1 || colordata>num_variables){
      cerr << "colordata must be between 1 and " << num_variables << ".\n";
      abort();
    }
    if (radius == 0)
      radius = ((maxs[0]-mins[0])*(maxs[1]-mins[1])*(maxs[2]-mins[2]))/(num_particles*2);
    radius*=radius_factor;
    
    if (debug) cerr << "Allocate Gridspheres  ";
    GridSpheres* grid = new GridSpheres(data, mins, maxs,
	num_particles, num_variables-3, gridcellsize,
	griddepth, radius, material_properties.size(), &material_properties[0]);
    display->attach(grid);
    timeblock->add(grid);
    if (debug) {
      cerr << "...Allocated Gridspheres\n";
      for(int m = 0; m < material_data_list.size(); m++) {
	MaterialData md = material_data_list[m];
	//vector<ParticleVariable<double> > pv_double_list;
	//vector<ParticleVariable<SCIRun::Point> > pv_point_list;
	//vector<ParticleVariable<SCIRun::Vector> > pv_vector_list;
	//ParticleVariable<SCIRun::Point> p_x;
	cerr << "md :" << m << endl;
	cerr << "PVD:" << md.pv_double_list.size() << endl;
	cerr << "PVV:" << md.pv_vector_list.size() << endl;
	cerr << "PVP:" << md.pv_point_list.size() << endl;
	cerr << "PV :" << md.p_x.getParticleSubset()->numParticles() << endl;
      }
    }
  }
  AuditDefaultAllocator();
  if (debug) cerr << "End of processdata\n";
}

#else
void append_spheres(rtrt::Array1<SphereData> &data_group,
		    vector<MaterialData> material_data_list,
		    const Uintah::Patch* patch,
		    bool do_PTvar_all, bool do_patch, bool do_material,
		    bool do_verbose, float radius, float radius_factor) {
  AuditDefaultAllocator();
  if (debug) cerr << "Start of append_spheres\n";
  if (debug) cerr << "Size of material_data_list = " << material_data_list.size() << "\n";
  if (material_data_list.size() <= 0)
    return;
  //--------------------------------------------------
  // set up the first min/max
  // and determine number of particles and variables
  SCIRun::Point min, max;
  vector<double> d_min,d_max,i_min,i_max,v_min,v_max,m_min,m_max;
  bool data_found = false;
  long num_particles = 0;
  int num_variables = 0;
  
  // loops until a non empty material_data set has been
  // found and inialized the mins and maxes
  for(int m = 0; m < material_data_list.size(); m++) {
    MaterialData md = material_data_list[m];
    if (debug) {
      cerr << "md.material_index = " << md.material_index << "\n";
      cerr << "md.pv_double_list.size() = " <<md.pv_double_list.size() << "\n";
      cerr << "md.pv_int_list.size() = " <<md.pv_int_list.size() << "\n";
      cerr << "md.pv_point_list.size() = " << md.pv_point_list.size() << "\n";
      cerr << "md.pv_vector_list.size() = " << md.pv_vector_list.size() <<"\n";
      cerr << "md.pv_matrix_list.size() = " << md.pv_matrix_list.size() <<endl;
    }
    ParticleSubset* pset = md.p_x.getParticleSubset();
    // pset is good because we already tested it before we passed
    // material_data_list into this function
    int numParticles = pset->numParticles();
    if(numParticles > 0){
      num_particles+=numParticles;
      ParticleSubset::iterator iter = pset->begin();
      
      if (!data_found) {
	// setup for p.x
	min=max=md.p_x[*iter];
	num_variables += 3;
	// setup for all others
	if (do_PTvar_all) {
	  for(int i = 0; i < md.pv_double_list.size(); i++) {
	    d_min.push_back(md.pv_double_list[i][*iter]);
	    d_max.push_back(md.pv_double_list[i][*iter]);
	    num_variables++;
	  }
	  for(int i = 0; i < md.pv_int_list.size(); i++) {
	    i_min.push_back(md.pv_int_list[i][*iter]);
	    i_max.push_back(md.pv_int_list[i][*iter]);
	    num_variables++;
	  }
	  for(int i = 0; i < md.pv_vector_list.size(); i++) {
	    v_min.push_back(md.pv_vector_list[i][*iter].length());
	    v_max.push_back(md.pv_vector_list[i][*iter].length());
	    num_variables++;
	  }
	  for(int i = 0; i < md.pv_matrix_list.size(); i++) {
	    m_min.push_back(md.pv_matrix_list[i][*iter].Norm());
	    m_max.push_back(md.pv_matrix_list[i][*iter].Norm());
	    num_variables++;
	  }
	}
	// initialized mins/maxes
	data_found = true;
      }
    } // end if(numParticles > 0)
  } // end material_data_list loop
  
  if (do_patch)
    num_variables++;
  if (do_material)
    num_variables++;
  if (do_verbose) {
    cerr << "Number of variables found: " << num_variables << endl;
    cerr << "Number of particles found in patch: " << num_particles << endl;
  }
  //--------------------------------------------------
  // extract data and write it to a file MaterialData at a time
  
  //---------
  // allocate memory for particle data
  SphereData sphere_data;
  sphere_data.data = (float*)malloc(num_variables*num_particles*sizeof(float));
  //	    if (do_verbose)
  //cerr << "---Extracting data and writing it out  ";
  float* p=sphere_data.data;
  if (do_verbose) cerr << "Past data allocation\n";
  for(int m = 0; m < material_data_list.size(); m++) {
    MaterialData md = material_data_list[m];
    ParticleSubset* pset = md.p_x.getParticleSubset();
#if 0 // don't need to do this anymore, because all the elements of
    // material_data_list have been checked already before passed in
    // a little redundant, but may not have been cought
    // by the previous section
    if (!pset) {
      cerr << "No particle location variable found\n";
      abort();
    }
#endif
    
    int numParticles = pset->numParticles();
    if (do_verbose) cerr << "m = " << m << "  numParticles = " << numParticles << endl;
    //total_particles+= numParticles;
    if(numParticles > 0){
      ParticleSubset::iterator iter = pset->begin();
      for(;iter != pset->end(); iter++){
	// p_x
	min=Min(min, md.p_x[*iter]);
	max=Max(max, md.p_x[*iter]);
	float temp_value = (float)(md.p_x[*iter]).x();
	*p++=temp_value;
	//fwrite(&temp_value, sizeof(float), 1, datafile);
	temp_value = (float)(md.p_x[*iter]).y();
	*p++=temp_value;
	
	//fwrite(&temp_value, sizeof(float), 1, datafile);
	temp_value = (float)(md.p_x[*iter]).z();
	*p++=temp_value;
	//fwrite(&temp_value, sizeof(float), 1, datafile);
	if (do_PTvar_all) {
	  // double data
	  for(int i = 0; i < md.pv_double_list.size(); i++) {
	    double value = md.pv_double_list[i][*iter];
	    d_min[i]=SCIRun::Min(d_min[i],value);
	    d_max[i]=SCIRun::Max(d_max[i],value);
	    temp_value = (float)value;
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  // int data
	  for(int i = 0; i < md.pv_int_list.size(); i++) {
	    double value = md.pv_int_list[i][*iter];
	    i_min[i]=SCIRun::Min(i_min[i],value);
	    i_max[i]=SCIRun::Max(i_max[i],value);
	    temp_value = (float)value;
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  // vector data
	  for(int i = 0; i < md.pv_vector_list.size(); i++) {
	    double value = md.pv_vector_list[i][*iter].length();
	    v_min[i]=SCIRun::Min(v_min[i],value);
	    v_max[i]=SCIRun::Max(v_max[i],value);
	    temp_value = (float)value;
	    *p++=temp_value;
	    
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  // Matrix data
	  for(int i = 0; i < md.pv_matrix_list.size(); i++) {
	    double value = md.pv_matrix_list[i][*iter].Norm();
	    m_min[i]=SCIRun::Min(m_min[i],value);
	    m_max[i]=SCIRun::Max(m_max[i],value);
	    temp_value = (float)value;
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  if (do_patch) {
	    temp_value = (float)patch->getID();
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	  if (do_material) {
	    temp_value = (float)m;
	    *p++=temp_value;
	    //fwrite(&temp_value, sizeof(float), 1, datafile);
	  }
	}
      }
    }
  }
  
  AuditDefaultAllocator();
  if (debug) cerr << "Past data write, on to datasummary.\n";
  //--------------------------------------------------
  // setup data summary and create GridSphere
  sphere_data.mins.resize(num_variables);
  sphere_data.maxs.resize(num_variables);
  int v = 0;
  if (debug) cerr << "data_found = " << data_found << "\n";
  if (data_found) {
    sphere_data.mins[0] = min.x();
    sphere_data.mins[1] = min.y();
    sphere_data.mins[2] = min.z();
    sphere_data.maxs[0] = max.x();
    sphere_data.maxs[1] = max.y();
    sphere_data.maxs[2] = max.z();
    v = 3;
    if (do_PTvar_all) {
      for(int i = 0; i < d_min.size(); i++) {
	sphere_data.mins[v] = d_min[i];
	sphere_data.maxs[v] = d_max[i];
	v++;
      }
      for(int i = 0; i < i_min.size(); i++) {
	sphere_data.mins[v] = i_min[i];
	sphere_data.maxs[v] = i_max[i];
	v++;
      }
      for(int i = 0; i < v_min.size(); i++) {
	sphere_data.mins[v] = v_min[i];
	sphere_data.maxs[v] = v_max[i];
	v++;
      }
      for(int i = 0; i < m_min.size(); i++) {
	sphere_data.mins[v] = m_min[i];
	sphere_data.maxs[v] = m_max[i];
	v++;
      }
      if (do_patch) {
	sphere_data.mins[v] = (float)patch->getID();
	sphere_data.maxs[v] = (float)patch->getID();
	v++;
      }
      if (do_material) {
	sphere_data.mins[v] = 0;
	sphere_data.maxs[v] = (float)material_data_list.size() ;
      }
    }

    if (radius == 0) {
      sphere_data.radius = ((sphere_data.maxs[0]-sphere_data.mins[0])*
			    (sphere_data.maxs[1]-sphere_data.mins[1])*
			    (sphere_data.maxs[2]-sphere_data.mins[2]))/
	                    (num_particles*2);
    } else {
      sphere_data.radius = radius;
    }
    sphere_data.radius *= radius_factor;
    
    sphere_data.nspheres = num_particles;
    sphere_data.numvars = num_variables;

    if (debug) {
      cerr << "...appended spheres\n";
      for(int m = 0; m < material_data_list.size(); m++) {
	MaterialData md = material_data_list[m];
	//vector<ParticleVariable<double> > pv_double_list;
	//vector<ParticleVariable<SCIRun::Point> > pv_point_list;
	//vector<ParticleVariable<SCIRun::Vector> > pv_vector_list;
	//ParticleVariable<SCIRun::Point> p_x;
	cerr << "md :" << m << endl;
	cerr << "PVD:" << md.pv_double_list.size() << endl;
	cerr << "PVV:" << md.pv_vector_list.size() << endl;
	cerr << "PVP:" << md.pv_point_list.size() << endl;
	cerr << "PV :" << md.p_x.getParticleSubset()->numParticles() << endl;
      }
    }
  }
  data_group.add(sphere_data);
  AuditDefaultAllocator();
  if (debug) cerr << "End of append_spheres\n";
}

GridSpheres* create_GridSpheres(rtrt::Array1<SphereData> data_group,
				int colordata, int gridcellsize,
				int griddepth) {
  // need from the group
  // 1. total number of spheres
  // 2. make sure the numvars is the same
  // 3. average radius

  int total_spheres = 0;
  if (debug) cerr << "Size of data_group = " << data_group.size() << endl;
  int numvars = data_group[0].numvars;
  float radius = 0;
  for (int i = 0; i < data_group.size(); i++) {
    if (debug) cerr << "data_group[" << i << "].numvars = " << data_group[i].numvars << endl;
    total_spheres += data_group[i].nspheres;
    if (numvars != data_group[i].numvars) {
      cerr << "numvars does not match: numvars = " << numvars << ", data_group[i].numvars = " << data_group[i].numvars << " Goodbye!\n";
      abort();
    }
    radius += data_group[i].radius;
  }
  radius /= data_group.size();
  cout << "radius = " << radius << endl;
  
  if(colordata < 1 || colordata > numvars){
    cerr << "colordata must be between 1 and " << numvars << ".\n";
    abort();
  }

  float *mins, *maxs;
  mins = (float*)malloc(numvars * sizeof(float));
  maxs = (float*)malloc(numvars * sizeof(float));
  // initialize the mins and maxs
  for (int i = 0; i < numvars; i++) {
    mins[i] =  MAXFLOAT;
    maxs[i] = -MAXFLOAT;
  }
  // now concatenate the spheres and compute the mins and maxs
  for (int i = 0; i < numvars; i++) {
    mins[i] =  MAXFLOAT;
    maxs[i] = -MAXFLOAT;
  }

  // allocate memory for the data
  static const int d_mem = 512;
  static const int nsph = (512*400);
  char* odata=(char*)malloc(numvars*(total_spheres+nsph)*sizeof(float)+d_mem);
  unsigned long addr=(unsigned long)odata;
  unsigned long off=addr%d_mem;
  if(off){
    addr+=d_mem-off;
  }

  // the data
  float *data = (float*)addr;
  // the index to the data array
  int index = 0;
  
  for (int g = 0; g < data_group.size(); g++) {
    // compute the mins and maxs
    for (int i = 0; i < numvars; i++) {
      mins[i] = Min(mins[i], data_group[g].mins[i]);
      maxs[i] = Max(maxs[i], data_group[g].maxs[i]);
    }
    // copy the data
    // this may be done more efficient using mcopy or something like it.
    int ndata = data_group[g].nspheres * numvars;
    for (int j = 0; j < ndata; j++) {
      data[index++] = data_group[g].data[j];
    }
  }
  if (index != total_spheres * numvars) {
    cerr << "Wrong number of vars copied: index = " << index << ", total_spheres * numvars = " << total_spheres * numvars << endl;
  }
  rtrt::Array1<Material*> matls;
  get_material(matls);
  return new GridSpheres(data, mins, maxs, total_spheres, numvars-3, gridcellsize, griddepth, radius, matls.size(), &matls[0]);  
}
#endif // ifdef PATCH_BY_PATCH


extern "C" 
Scene* make_scene(int argc, char* argv[], int /*nworkers*/)
{
  //------------------------------
  // Default values
  bool do_PTvar = true;
  bool do_PTvar_all = true;
  bool do_patch = false;
  bool do_material = false;
  bool do_verbose = false;
  int time_step_lower = -1;
  int time_step_upper = -1;
  int gridcellsize=4;
  int griddepth=1;
  int colordata=2;
  float radius_factor=1;
  float rate=3;
  float radius=0;
  string filebase;

  //------------------------------
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s=argv[i];
    cerr << "Parsing argument : " << s << endl;
    if(s == "-PTvar") {
      do_PTvar = true;
    } else if (s == "-ptonly") {
      do_PTvar_all = false;
    } else if (s == "-patch") {
      do_patch = true;
    } else if (s == "-material") {
      do_material = true;
    } else if (s == "-verbose") {
      do_verbose = true;
    } else if (s == "-debug") {
      debug = true;
    } else if (s == "-timesteplow") {
      time_step_lower = atoi(argv[++i]);
    } else if (s == "-timestephigh") {
      time_step_upper = atoi(argv[++i]);
    } else if(s == "-gridcellsize") {
      i++;
      gridcellsize=atoi(argv[i]);
    } else if(s == "-griddepth") {
      i++;
      griddepth=atoi(argv[i]);
    } else if(s == "-colordata") {
      i++;
      colordata=atoi(argv[i]);
    } else if(s == "-radiusfactor") {
      i++;
      radius_factor=atof(argv[i]);
    } else if(s == "-radius") {
      i++;
      radius=atof(argv[i]);
    } else if(s == "-rate"){
      i++;
      rate=atof(argv[i]);
    } else if( (s == "-help") || (s == "-h") ) {
      usage( "", argv[0] );
      return(0);
    } else {
      if(filebase!="") {
	usage(s, argv[0]);
	return(0);
      }
      else
	filebase = argv[i];
    }
  }
  
  //------------------------------
  // setup variales/states
  if(filebase == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
    return(0);
  }

  try {
    XMLPlatformUtils::Initialize();
  } catch(const XMLException& toCatch) {
    cerr << "Caught XML exception: " << toString(toCatch.getMessage()) 
	 << '\n';
    exit( 1 );
  }
  
  
  Group* all = new Group();
  // the value of colordata will be checked later and the
  // program will abort if the value is too large.
  GridSpheresDpy* display = new GridSpheresDpy(colordata-1);
  TimeObj* alltime = new TimeObj(rate);

  // setup colors
  CatmullRomSpline<Color> spline(0);
  spline.add(Color(.4,.4,.4));
  spline.add(Color(.4,.4,1));
  spline.add(Color(.4,1,.4));
  spline.add(Color(1,1,.4));
  spline.add(Color(1,.4,.4));
  int ncolors=5000;
  rtrt::Array1<Material*> material_properties(ncolors);
  float Ka=.8;
  float Kd=.8;
  float Ks=.8;
  float refl=0;
  float specpow=40;
  for(int i=0;i<ncolors;i++){
    float frac=float(i)/(ncolors-1);
    Color c(spline(frac));
    material_properties[i]=new Phong(c*Ka, c*Kd, c*Ks, specpow, refl);
    //material_properties[i]=new LambertianMaterial(c*Kd);
  }

  
  try {
    DataArchive* da = new DataArchive(filebase);

    cerr << "Done parsing dataArchive\n";
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;
    da->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    cout << "There are " << vars.size() << " variables:\n";
    
    vector<int> index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    cout << "There are " << index.size() << " timesteps:\n";
    
    //------------------------------
    // figure out the lower and upper bounds on the timesteps
    if (time_step_lower <= -1)
      time_step_lower =0;
    else if (time_step_lower >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      abort();
    }
    if (time_step_upper <= -1)
      time_step_upper = times.size()-1;
    else if (time_step_upper >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      abort();
    }

    //------------------------------
    // start the data extraction
    
    // data structure for all the spheres
    rtrt::Array1<SphereData> sphere_data;
      
    // for all timesteps
    for(int t=time_step_lower;t<=time_step_upper;t++){
      AuditDefaultAllocator();	  
      if (debug) cerr << "Started timestep\n";

      sphere_data.remove_all();
      
      double time = times[t];
      GridP grid = da->queryGrid(time);
      if(do_verbose)
	cout << "time = " << time << "\n";
      cerr << "Creating new timeblock.\n";
#ifdef PATCH_BY_PATCH
      Group* timeblock = new Group();
#endif
      //Group* timeblock2 = new Group();
      
      // for each level in the grid
      for(int l=0;l<grid->numLevels();l++){
	AuditDefaultAllocator();	  
	if (debug) cerr << "Started level\n";
	LevelP level = grid->getLevel(l);
	
	// for each patch in the level
	for(Level::const_patchIterator iter = level->patchesBegin();
	    iter != level->patchesEnd(); iter++){
#if 0
	  AuditDefaultAllocator();	  
	  if (debug) cerr << "Started patch\n";
#endif
	  const Patch* patch = *iter;
	  vector<MaterialData> material_data_list; 
	  
	  // for all vars in one timestep in one patch
	  for(int v=0;v<vars.size();v++){
	    std::string var = vars[v];
	    if (do_verbose) cerr << "var = " << var << endl;
	    const Uintah::TypeDescription* td = types[v];
	    const Uintah::TypeDescription* subtype = td->getSubType();
	    //---------int numMatls = da->queryNumMaterials(var, patch, time);
	    ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
	    
	    // for all the materials in the patch belonging to the variable
	    for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
		matlIter != matls.end(); matlIter++){
	      int matl = *matlIter;
	      //------for(int matl=0;matl<numMatls;matl++){
	      MaterialData material_data;
	      // the index into material_data_list that corresponds to the
	      // material
	      int matl_index = -1;
	      
	      // we only have one material_data for each material reguardless
	      // of what variable it is associated with it.  Therefore you
	      // need to determine if you can use an already created one or if
	      // you need to create one.
	      if (debug) cerr << "matl = " << matl << ", material_data_list.size() = " << material_data_list.size() << endl;
	      for (unsigned int m = 0; m < material_data_list.size(); m++) {
		if (debug) cerr << "m = " << m << ", material_index = " << material_data_list[m].material_index << endl;
		if (matl == material_data_list[m].material_index) {
		  matl_index = m;
		  material_data = material_data_list[m];
		  break;
		}
	      }

#if 0
	      cerr << "Var " << var << "  ";
	      cerr << "Uintah::TypeDescription::ParticleVariable" << Uintah::TypeDescription::ParticleVariable << endl;
#endif
	      switch(td->getType()){
	      case Uintah::TypeDescription::ParticleVariable:
		if (do_PTvar) {
		  switch(subtype->getType()){
		  case Uintah::TypeDescription::double_type:
		    {
		      ParticleVariable<double> value;
		      da->query(value, var, matl, patch, time);
		      material_data.pv_double_list.push_back(value);
		    }
		  break;
		  case Uintah::TypeDescription::int_type:
		    {
		      ParticleVariable<int> value;
		      da->query(value, var, matl, patch, time);
		      material_data.pv_int_list.push_back(value);
		    }
		  break;
		  case Uintah::TypeDescription::Point:
		    {
		      ParticleVariable<SCIRun::Point> value;
		      da->query(value, var, matl, patch, time);
		      
		      if (var == "p.x") {
			if (debug) cerr << "Found p.x" << endl;
			material_data.p_x = value;
		      } else {
			material_data.pv_point_list.push_back(value);
		      }
		    }
		  break;
		  case Uintah::TypeDescription::Vector:
		    {
		      ParticleVariable<SCIRun::Vector> value;
		      da->query(value, var, matl, patch, time);
		      material_data.pv_vector_list.push_back(value);
		    }
		  break;
  		  case Uintah::TypeDescription::Matrix3:
 		    {
  		      ParticleVariable<Matrix3> value;
  		      da->query(value, var, matl, patch, time);
  		      material_data.pv_matrix_list.push_back(value);
  		    }
		  break;
		  default:
		    cerr << "Particle Variable of unknown type: " << subtype->getType() << '\n';
		    break;
		  }
		  break;
		}
		break;
	      case Uintah::TypeDescription::NCVariable:
	      case Uintah::TypeDescription::CCVariable:
	      case Uintah::TypeDescription::Matrix3:
	      case Uintah::TypeDescription::ReductionVariable:
	      case Uintah::TypeDescription::Unknown:
	      case Uintah::TypeDescription::Other:
		// currently not implemented
		break;
	      default:
		cerr << "Variable (" << var << ") is of unknown type: " << td->getType() << '\n';
		break;
	      } // end switch(td->getType())
	      if (matl_index != -1) {
		material_data_list[matl_index] = material_data;
	      } else {
		material_data.material_index = matl;
		material_data_list.push_back(material_data);
	      }
	    } // end matl
	    
	  } // end vars
	  // after all the variable data has been collected write it out
	  if (do_PTvar) {
	    if (debug) cerr << "Processing extracted data\n";

	    // remove and material_data's that don't have a p.x
	    for(vector<MaterialData>::iterator iter=material_data_list.begin();
		  iter != material_data_list.end();) {
	      MaterialData md = *iter;
	      ParticleSubset* pset = md.p_x.getParticleSubset();
	      if (debug) cerr << "numParticles = " << pset->numParticles() << endl;
	      if ((pset == NULL) || (pset->numParticles() <= 0)) {
		if (do_verbose) cerr << "No particle location variable found or no particles exist for material" << md.material_index << " over patch " << patch->getID() << "\n";
		iter = material_data_list.erase(iter);
		if (do_verbose) cerr << "Size of material_data_list is " << material_data_list.size() << endl;
	      } else {
		iter++;
	      }
	    }
#ifdef PATCH_BY_PATCH
	    processdata(material_data_list, patch,
		 do_PTvar_all, do_patch, do_material,
		 do_verbose, colordata, radius,
		 radius_factor, display, timeblock, material_properties,
		 gridcellsize, griddepth);
#else
	    append_spheres(sphere_data, material_data_list, patch,
			   do_PTvar_all, do_patch, do_material,
			   do_verbose, radius, radius_factor);
#endif
	    if (debug) cerr << "Finished processdata(1)\n";
	  }
	  AuditDefaultAllocator();	  
	  if (debug) cerr << "Finished processdata\n";
	} // end patch
	AuditDefaultAllocator();	  
	if (debug) cerr << "Finished patch\n";
      } // end level
      AuditDefaultAllocator();	  
      if (debug) cerr << "Finished level\n";
      cout << "Adding timestep.\n";
      //Material* matl0=new Phong(Color(0,0,0), Color(.2,.2,.2), Color(.3,.3,.3), 10, .5);
      //timeblock2->add(new Sphere(matl0,::Point(t,t,t),1));
      //alltime->add(timeblock2);
#ifdef PATCH_BY_PATCH
      alltime->add(timeblock);
#else
      GridSpheres* obj = create_GridSpheres(sphere_data,colordata,
					    gridcellsize,griddepth);
      display->attach(obj);
      alltime->add((Object*)obj);
#endif
    } // end timestep
    all->add(alltime);
    AuditDefaultAllocator();	  
    if (debug) cerr << "Finished timestep\n";
  } catch (SCIRun::Exception& e) {
    cerr << "Caught exception: " << e.message() << '\n';
    abort();
  } catch(...){
    cerr << "Caught unknown exception\n";
    abort();
  }

  AuditDefaultAllocator();	  
  if (debug) cerr << "Finished processdata/patch\n";
  if (debug) cerr << "Creating GridSpheres display thread\n";
  //#if 0
  new Thread(display, "GridSpheres display thread\n");
  //#endif
#if 0
  Group *all2 = new Group();
  TimeObj* alltime2 = new TimeObj(rate);
  Group* timeblock2;
  for (int t = 0; t < 5; t++) {
    timeblock2 = new Group();
    Material* matl0=new Phong(Color(1,0,0), Color(.5,.2,.2), Color(.8,.3,.3), 10, .5);
    timeblock2->add((Object*)new Sphere(matl0,::Point(t,t,t),1));
    alltime2->add((Object*)timeblock2);
  }
  all2->add((Object*)alltime2);
#endif
  
  rtrt::Plane groundplane (rtrt::Point(-500, 300, 0), rtrt::Vector(7, -3, 2));
  Camera cam(rtrt::Point(0,0,400),rtrt::Point(0,0,0),rtrt::Vector(0,1,0),60.0);
  double bgscale=0.5;
  Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);
  double ambient_scale=1.0;
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  Scene* scene=new Scene(all, cam,
			 bgcolor, cdown, cup, groundplane, 
			 ambient_scale);
  
  scene->add_light(new Light(rtrt::Point(500,-300,300), Color(.8,.8,.8), 0));
  scene->shadow_mode=1;
  return scene;
}















