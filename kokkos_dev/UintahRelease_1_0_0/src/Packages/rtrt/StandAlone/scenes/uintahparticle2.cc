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
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Sphere.h>
#include <Packages/rtrt/Core/SelectableGroup.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/RegularColorMap.h>
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
#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

// For threaded extraction of particles
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Semaphore.h>

#include <Core/XMLUtil/XMLUtil.h>
// general

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <sgi_stl_warnings_on.h>

#include <sci_values.h>


using namespace std;
using namespace Uintah;
using namespace rtrt;

bool debug = false;

bool write_data = false;

// don't know if this is the way to do it --bigler
extern "C" void AuditDefaultAllocator() {
}

extern "C" void audit() {
  AuditDefaultAllocator();
}

Mutex data_allocator("Data Allocator");
unsigned long long total_allocated = 0;
unsigned long long total_freed = 0;

class SphereData {
public:
  float* data;
  long nspheres;
  int numvars;
  float radius;
  string *var_names;
};

class MaterialData {
public:
  MaterialData():material_number(-1), ndata(0), data(0) {}
  MaterialData(int mat_num, long ndata):
    material_number(mat_num), ndata(ndata) {
    data = (float*)malloc(sizeof(float)*ndata);
  }
  ~MaterialData() {}

  void deleteme() {
    if (data)
      free(data);
  }
  
  int material_number;
  long ndata;
  float *data;
};

class VariableData {
public:
  string name;
  vector<MaterialData> material_set;

  void print() {
    cerr << "name = " << name << ", size = " << material_set.size();
  }

  void deleteme() {
    for(unsigned int i = 0; i < material_set.size(); i++)
      material_set[i].deleteme();
  }
};

class PatchData {
public:
  VariableData position_x;
  VariableData position_y;
  VariableData position_z;
  vector< VariableData > variables;

  void print() {
    cerr << "position_x:"; position_x.print(); cerr << endl;
    cerr << "position_y:"; position_y.print(); cerr << endl;
    cerr << "position_z:"; position_z.print(); cerr << endl;
    for(unsigned int i = 0; i < variables.size(); i++) {
      cerr << "var(" << i << "):"; variables[i].print(); cerr << endl;
    }
  }

  void deleteme() {
    position_x.deleteme();
    position_y.deleteme();
    position_z.deleteme();
    for (unsigned int i = 0; i < variables.size(); i++)
      variables[i].deleteme();
  }
};

void usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
        cerr << "Error parsing argument: " << badarg << '\n';
    cerr << "Usage: " << progname << " [options] <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h[elp]\n";
    cerr << "  -cmap [file with rgb triples]\n";
    cerr << "  -cmaptype <type>     type can be several strings.\n";
    cerr << "  -ptonly (outputs only the point locations\n";
    cerr << "  -patch (outputs patch id with data)\n";
    cerr << "  -material (outputs material number with data)\n";
    cerr << "  -i,--include <var> include this variable\n";
    cerr << "                     (excludes all non included variables)\n";
    cerr << "  -verbose (prints status of output)\n";
    cerr << "  -timestep [int] (output only a single given timestep)\n";
    cerr << "  -timesteplow [int] (only outputs timestep from int)\n";
    cerr << "  -timestephigh [int] (only outputs timesteps upto int)\n";
    cerr << "  -timestepinc [int] (use only every nth timestep)\n";
    cerr << "  -gridcellsize [int]\n";
    cerr << "  -griddepth [int]\n";
    cerr << "  -colordata [int]\n";
    cerr << "  -radiusfactor [float]\n";
    cerr << "  -radius [float]\n";
    cerr << "  -radius_index [int]\n";
    cerr << "  -radius_from_volume (if p.volume is present create p.radius)\n";
    cerr << "  -rate [float]\n";
    cerr << "  -dpyconfig [filename] file used to configure the display\n";
    cerr << "  -patches [int] only read in the first int non empty\n";
    cerr << "                 patches of particles\n";
    cerr << "  -writedata     outputs files to uintahout/spheredata<timestep>.raw\n";
    return;
}

void writeData(float *spheres, int nspheres, int ndata, int tindex) {
  char buf[200];
  sprintf(buf, "uintahout/spheredata%03d.raw", tindex);
  cerr << "Writing out dataset to "<<buf<<"\n";
  FILE *out = fopen(buf, "wb");
  if (!out) {
    cerr << "Could not open "<<buf<<" for writing.\n";
    return;
  }
  int nvalues = nspheres * ndata;
  size_t wrote = fwrite(spheres, sizeof(float), nvalues, out);
  if (wrote != nvalues) {
    cerr << "Only wrote out "<<wrote<<" floats instead of "<<nvalues<<"\n";
    return;
  }
  fclose(out);
  
  cerr << "Done writing out dataset\n";
}

void append_spheres(rtrt::Array1<SphereData> &data_group,
                    PatchData patchdata,
                    const Uintah::Patch* patch,
                    bool do_PTvar_all, bool do_patch, bool do_material,
                    bool do_verbose, float radius, float radius_factor) {
  if (debug) cerr << "Start of append_spheres\n";
  if (debug) patchdata.print();
  //  bool data_found = false;
  long num_particles = 0;
  unsigned int num_variables = 0;
  
  // calculate the number of particles and number of variables

  // all materials must have a position variable
  unsigned long num_materials = patchdata.position_x.material_set.size();
  if (do_verbose) cerr << "Number of materials is " << num_materials << endl;
  if (num_materials == 0)
    return;
  // count the number of variables
  num_variables += 3; // three for the position
  if (do_PTvar_all) {
    num_variables += patchdata.variables.size();
  }
  if (do_patch)
    num_variables++;
  if (do_material)
    num_variables++;
  // count the number of particles
  for (unsigned int i = 0; i < num_materials; i++) {
    num_particles += patchdata.position_x.material_set[i].ndata;
  }
  
  if (do_verbose) {
    cerr << "Number of variables found: " << num_variables << endl;
    cerr << "Number of particles found in patch: " << num_particles << endl;
  }
  if (num_particles == 0)
    return;
  //--------------------------------------------------
  // extract data and write it to spheredata material at a time
  
  //---------
  // allocate memory for particle data
  SphereData sphere_data;
  size_t num_bytes = num_variables*num_particles*sizeof(float);
  sphere_data.data = (float*)malloc(num_variables*num_particles*sizeof(float));
  data_allocator.lock(); total_allocated += num_bytes; data_allocator.unlock();
  //cerr << "Allocated "<<num_bytes<<" of memory to "<<sphere_data.data<<endl;
  float* p=sphere_data.data;
  if (do_verbose) cerr << "Past data allocation\n";
  // Loop over each material set for the position variables.
  // 1. If there are no particles over a particular material skip it.
  // 2. If a variable does not exist over a particular data set, pad it
  //    with zeros.
  float patchid = (float)patch->getID();
  for(unsigned int matind = 0; matind < num_materials; matind++) {
    long num_parts = patchdata.position_x.material_set[matind].ndata;
    if (num_parts > 0) {
      // alrighty, we have a non empty particle material
      MaterialData pos_x = patchdata.position_x.material_set[matind];
      MaterialData pos_y = patchdata.position_y.material_set[matind];
      MaterialData pos_z = patchdata.position_z.material_set[matind];
      int matl = pos_x.material_number;
      for(long partind = 0; partind < num_parts; partind++) {
        *p++ = pos_x.data[partind];
        *p++ = pos_y.data[partind];
        *p++ = pos_z.data[partind];
        if (do_PTvar_all) {
          for(unsigned int i = 0; i < patchdata.variables.size(); i++) {
            int var_matind = -1;
            // now look for the matching material_number
            vector<MaterialData> mat_set = patchdata.variables[i].material_set;
            for(unsigned int j = 0; j < mat_set.size(); j++) {
              if (mat_set[j].material_number == matl) {
                // we've found it
                var_matind = j;
                break;
              }
            }
            if (var_matind >= 0)
              *p++ = mat_set[var_matind].data[partind];
            else
              // pad the data
              *p++ = 0;
          }
        }
        if (do_patch)
          *p++ = patchid;
        if (do_material)
          *p++ = (float)matl;
      } // end for(num_parts)
    } // end if(num_parts > 0)
  } // end for(num_materials)
  
  AuditDefaultAllocator();
  if (debug) cerr << "Past data write, on to datasummary.\n";
  if (radius == 0) {
    radius = 1;
  } else {
    sphere_data.radius = radius;
  }
  sphere_data.radius *= radius_factor;
    
  sphere_data.nspheres = num_particles;
  sphere_data.numvars = num_variables;

  if (debug) {
    cerr << "...appended spheres\n";
  }
  sphere_data.var_names = new string[num_variables];
  unsigned int n_index = 0;
  sphere_data.var_names[n_index++] = patchdata.position_x.name;
  sphere_data.var_names[n_index++] = patchdata.position_y.name;
  sphere_data.var_names[n_index++] = patchdata.position_z.name;
  for (unsigned int i = 0; i < patchdata.variables.size(); i++)
    sphere_data.var_names[n_index++] = patchdata.variables[i].name;
  if (do_patch)
    sphere_data.var_names[n_index++] = string("Patch ID");
  if (do_material)
    sphere_data.var_names[n_index++] = string("Material Index");
  if (n_index != num_variables)
    cerr << "n_index = " << n_index<<", num_variables = "<<num_variables<<endl;
  data_group.add(sphere_data);
  AuditDefaultAllocator();
  if (debug) cerr << "End of append_spheres\n";
}

GridSpheres* create_GridSpheres(rtrt::Array1<SphereData> data_group,
                                int colordata, int gridcellsize,
                                int griddepth, RegularColorMap *cmap,
                                int timestep) {
  // need from the group
  // 1. total number of spheres
  // 2. make sure the numvars is the same
  // 3. average radius

  int total_spheres = 0;
  if (debug) cerr << "Size of data_group = " << data_group.size() << endl;
  if (data_group.size() == 0) {
    cerr << "No particles. Exiting\n";
    Thread::exitAll(0);
  }
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
    // copy the data
    // this may be done more efficient using mcopy or something like it.
    long ndata = data_group[g].nspheres * numvars;
    for (long j = 0; j < ndata; j++) {
      data[index++] = data_group[g].data[j];
    }
    // Delete the memory we don't use anymore
    if (data_group[g].data) free(data_group[g].data);
    size_t num_bytes = data_group[g].nspheres * numvars * sizeof(float);
    data_allocator.lock(); total_freed += num_bytes; data_allocator.unlock();
  }
  if (index != total_spheres * numvars) {
    cerr << "Wrong number of vars copied: index = " << index << ", total_spheres * numvars = " << total_spheres * numvars << endl;
  }
  
  float *mins, *maxs;
  mins = (float*)malloc(numvars * sizeof(float));
  maxs = (float*)malloc(numvars * sizeof(float));
  // initialize the mins and maxs
  for (int i = 0; i < numvars; i++) {
    mins[i] =  FLT_MAX;
    maxs[i] = -FLT_MAX;
  }

  cerr << "Total number of spheres: " << total_spheres << endl;
  if (write_data) {
    writeData(data, total_spheres, numvars, timestep);
  }
  return new GridSpheres(data, mins, maxs, total_spheres, numvars, gridcellsize, griddepth, radius, cmap, data_group[0].var_names);  
}

class Preprocessor: public Runnable {
  GridSpheres *grid;
  Semaphore *sema;

public:
  Preprocessor(GridSpheres* grid, Semaphore *sema):
    grid(grid), sema(sema)
  {}
  ~Preprocessor() {}
  void run() {
    int a,b;
    grid->preprocess(0,a,b);
    sema->up();
  }
};

class SphereMaker: public Runnable {
private:
  Mutex *amutex;
  Semaphore *sema;
  Patch *patch;
  DataArchive *da;
  rtrt::Array1<SphereData> *sphere_data_all;

  double time;
  vector<string> *vars;
  vector<string> *var_include;
  vector<const Uintah::TypeDescription*> *types;

  bool do_PTvar_all, do_patch, do_material, do_verbose, compute_radius;
  double radius, radius_factor;
public:
  SphereMaker(Mutex *amutex, Semaphore *sema, Patch *patch, DataArchive *da,
              rtrt::Array1<SphereData> *sda,
              double time, vector<string> *vars, vector<string> *var_include,
              vector<const Uintah::TypeDescription*> *types,
              bool do_PTvar_all, bool do_patch, bool do_material,
              bool do_verbose, bool compute_radius,
              double radius, double radius_factor):
    amutex(amutex), sema(sema), patch(patch), da(da), sphere_data_all(sda),
    time(time), vars(vars), var_include(var_include), types(types),
    do_PTvar_all(do_PTvar_all), do_patch(do_patch), do_material(do_material),
    do_verbose(do_verbose), compute_radius(compute_radius),
    radius(radius), radius_factor(radius_factor)
  {}
  ~SphereMaker() {}

  template<class PartT>
  void getParticleData(PatchData& patchdata, string& var,
                       ConsecutiveRangeSet& matls, Patch* patch, double time) {
    if (!do_PTvar_all) return;
    bool do_radius_computation = false;
    if (compute_radius && var == "p.volume") do_radius_computation = true;
    VariableData vardata;
    // If we are computing the radius, change the name
    if (!do_radius_computation)
      vardata.name = var;
    else
      vardata.name = "Radius from p.volume";
    for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
        matlIter != matls.end(); matlIter++){
      int matl = *matlIter;
                
      ParticleVariable<PartT> value;
      da->query(value, var, matl, patch, time);
      ParticleSubset* pset = value.getParticleSubset();
      if (!pset) return;
      int numParticles = pset->numParticles();
      if (numParticles > 0) {
        // extract the data
        MaterialData md(matl,numParticles);
        float *p = md.data;
        if (!do_radius_computation) {
          for(ParticleSubset::iterator iter = pset->begin();
              iter != pset->end(); iter++) {
            float temp_value= (float)value[*iter];
            *p++ = temp_value;
          }
        } else {
          for(ParticleSubset::iterator iter = pset->begin();
              iter != pset->end(); iter++) {
            float temp_value= (float)value[*iter];
            if (temp_value > 0)
              temp_value = powf(temp_value*(3.0f/4.0f*M_1_PI), 1.0f/3.0f);
            else
              temp_value = 0;
            *p++ = temp_value;
          }
        }
        // add the extracted data to the variable
        vardata.material_set.push_back(md);
      }
    } // end material loop
    // now this variable for this patch is completely extracted
    // put it in patchdata
    patchdata.variables.push_back(vardata);
  }

  
  void run() {
    PatchData patchdata;
    rtrt::Array1<SphereData> sphere_data;
    Matrix3 one; one.Identity();
    
    // for all vars in one timestep in one patch
    bool gotPos = false;
    for(unsigned int v=0;v<vars->size();v++){
      std::string var = (*vars)[v];
      if (var_include->size() > 0) {
        // Only do this check if the size of var_include is > 0.
        bool var_is_found = false;
        for(unsigned int s=0; s<var_include->size(); s++)
          if ((*var_include)[s] == var) {
            var_is_found = true;
            break;
          }
        // if the variable was not found in our little list
        if (!var_is_found) continue;
      }
      const Uintah::TypeDescription* td = (*types)[v];
      const Uintah::TypeDescription* subtype = td->getSubType();
      //---------int numMatls = da->queryNumMaterials(var, patch, time);
      ConsecutiveRangeSet matls = da->queryMaterials(var, patch, time);
      
      // now do something different depending on the type of the variable
      switch(td->getType()){
      case Uintah::TypeDescription::ParticleVariable:
        {
          switch(subtype->getType()){
          case Uintah::TypeDescription::double_type:
            getParticleData<double>(patchdata, var, matls, patch, time);
            break;
          case Uintah::TypeDescription::float_type:
            getParticleData<float>(patchdata, var, matls, patch, time);
            break;
          case Uintah::TypeDescription::int_type:
            getParticleData<int>(patchdata, var, matls, patch, time);
            break;
          case Uintah::TypeDescription::Point:
            {
              if (gotPos) {
                if (debug) cerr << "Already know position" << endl;
                break;
              }
              if (var == "p.x" || var == "p.xx") {
                if (debug) cerr << "Found p.x" << endl;
                gotPos = true;
              } else {
                // don't know how to handle a point variable that's not
                // p.x
                break;
              }
              VariableData position_x; position_x.name = "x";
              VariableData position_y; position_y.name = "y";
              VariableData position_z; position_z.name = "z";
              for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  matlIter != matls.end(); matlIter++){
                int matl = *matlIter;
                
                if (debug) cerr << "matl = " << matl << endl;
                ParticleVariable<SCIRun::Point> value;
                da->query(value, var, matl, patch, time);
                ParticleSubset* pset = value.getParticleSubset();
                if (!pset) break;
                int numParticles = pset->numParticles();
                if (numParticles > 0) {
                  if (debug) cerr<<"numParticles = "<<numParticles;
                  // x
                  MaterialData mdx(matl,numParticles);
                  float *px = mdx.data;
                  // y
                  MaterialData mdy(matl,numParticles);
                  float *py = mdy.data;
                  // z
                  MaterialData mdz(matl,numParticles);
                  float *pz = mdz.data;
                  // extract the data
                  int i=0;
                  for(ParticleSubset::iterator iter = pset->begin();
                      iter != pset->end(); iter++,i++) {
                    // x
                    float temp_value = (float)(value[*iter].x());
                    *px++ = temp_value;
                    // y
                    temp_value = (float)(value[*iter].y());
                    *py++ = temp_value;
                    // z
                    temp_value = (float)(value[*iter].z());
                    *pz++ = temp_value;
                  }
                  //if (debug) mdx.print();
                  if (debug) cerr<<", i = "<<i<<endl;
                  // add the extracted data to the variable
                  position_x.material_set.push_back(mdx);
                  position_y.material_set.push_back(mdy);
                  position_z.material_set.push_back(mdz);
                }
              } // end material loop
              // now this variable for this patch is completely extracted
              // put it in patchdata
              if (debug) { position_x.print(); cerr << endl; }
              patchdata.position_x = position_x;
              patchdata.position_y = position_y;
              patchdata.position_z = position_z;
            }
          break;
          case Uintah::TypeDescription::Vector:
            {
              if (!do_PTvar_all) break;
              VariableData vardata;
              vardata.name = string(var+" length");
              for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  matlIter != matls.end(); matlIter++){
                int matl = *matlIter;
                
                ParticleVariable<SCIRun::Vector> value;
                da->query(value, var, matl, patch, time);
                ParticleSubset* pset = value.getParticleSubset();
                if (!pset) break;
                int numParticles = pset->numParticles();
                if (numParticles > 0) {
                  // extract the data
                  MaterialData md(matl,numParticles);
                  float *p = md.data;
                  for(ParticleSubset::iterator iter = pset->begin();
                      iter != pset->end(); iter++) {
                    float temp_value= (float)(value[*iter].length());
                    *p++ = temp_value;
                  }
                  // add the extracted data to the variable
                  vardata.material_set.push_back(md);
                }
              } // end material loop
              // now this variable for this patch is completely extracted
              // put it in patchdata
              patchdata.variables.push_back(vardata);
            }
          break;
          case Uintah::TypeDescription::Matrix3:
            {
              if (!do_PTvar_all) break;
              VariableData vardata;
              VariableData vardata2;
              // can extract Determinant(), Trace(), Norm()
              vardata.name = string(var+" Trace/3");
              vardata2.name = string(var+" Equivalent");
              for(ConsecutiveRangeSet::iterator matlIter = matls.begin();
                  matlIter != matls.end(); matlIter++){
                int matl = *matlIter;
                
                ParticleVariable<Matrix3> value;
                da->query(value, var, matl, patch, time);
                ParticleSubset* pset = value.getParticleSubset();
                if (!pset) break;
                int numParticles = pset->numParticles();
                if (numParticles > 0) {
                  // extract the data
                  MaterialData md(matl,numParticles);
                  float *p = md.data;
                  MaterialData md2(matl,numParticles);
                  float *p2 = md2.data;
                  for(ParticleSubset::iterator iter = pset->begin();
                      iter != pset->end(); iter++) {
                    float temp_value= (float)(value[*iter].Trace()/3.0);
                    *p++ = temp_value;
                    temp_value= (float)(sqrt(1.5*(value[*iter]-one*temp_value).NormSquared()));
                    *p2++ = temp_value;
                  }
                  // add the extracted data to the variable
                  vardata.material_set.push_back(md);
                  vardata2.material_set.push_back(md2);
                }
              } // end material loop
              // now this variable for this patch is completely extracted
              // put it in patchdata
              patchdata.variables.push_back(vardata);
              patchdata.variables.push_back(vardata2);
            }
          break;
          default:
            cerr << "Particle Variable of unknown type: " << subtype->getName() << '\n';
            break;
          } // end switch(subtype)
        } // end case ParticleVariable
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
        cerr << "Variable (" << var << ") is of unknown type: " << td->getName() << '\n';
        break;
      } // end switch(td->getType())
      if (debug) cerr << "Finished var " << var << "\n";
    }
    AuditDefaultAllocator();
    append_spheres(sphere_data, patchdata, patch,
                   do_PTvar_all, do_patch, do_material,
                   do_verbose, radius, radius_factor);
    patchdata.deleteme();
    amutex->lock();
    for(int i = 0; i < sphere_data.size(); i++) {
      sphere_data_all->add(sphere_data[i]);
    }
    cerr << "Read Patch(" << patch->getID() << ")\n";
    amutex->unlock();
    sema->up();
  }
private:
  // member variables
  
};

extern "C" 
Scene* make_scene(int argc, char* argv[], int nworkers)
{
  //------------------------------
  // Default values
  bool do_PTvar_all = true;
  bool do_patch = false;
  bool do_material = false;
  bool do_verbose = false;
  bool compute_radius = false;
  int time_step_lower = -1;
  int time_step_upper = -1;
  int time_step_inc = 1;
  int gridcellsize=6;
  int griddepth=2;
  int colordata=2;
  float radius_factor=1;
  float rate=3;
  float radius=0;
  string filebase;
  int non_empty_patches = -1;
  char *dpy_config = 0;
  // Only use var_include if the size is greater than 0.
  vector<std::string> var_include;
  char *cmap_file = 0; // Non zero when a file has been specified
  char *cmap_type = "InvRainbow";
  // This is the index to use for the radius.  -1 means don't use it.
  int radius_index = -1;

  //------------------------------
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s=argv[i];
    cerr << "Parsing argument : " << s << endl;
    if (s == "-ptonly") {
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
    } else if (s == "-timestep") {
      // Set lower and upper to be the same value
      time_step_lower = time_step_upper = atoi(argv[++i]);
    } else if (s == "-timestepinc") {
      time_step_inc = atoi(argv[++i]);
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
    } else if(s == "-radius_index") {
      radius_index = atoi(argv[++i]);
    } else if(s == "-radius_from_volume") {
      compute_radius = true;
    } else if(s == "-rate"){
      i++;
      rate=atof(argv[i]);
    } else if (s == "-patches") {
      non_empty_patches = atoi(argv[++i]);
    } else if (s == "-dpyconfig") {
      i++;
      dpy_config = argv[i];
    } else if( (s == "-help") || (s == "--help") || (s == "-h") ) {
      usage( "", argv[0] );
      return(0);
    } else if (s == "-cmap") {
      cmap_file = argv[++i];
    } else if (s == "-cmaptype") {
      cmap_type = argv[++i];
    } else if (s == "-i" || (s == "--include")) {
      if (var_include.size() == 0) {
        // We are going to push back p.x right now, because we know we will
        // always need it.
        var_include.push_back("p.x");
      }
      i++;
      var_include.push_back(argv[i]);
    } else if (s == "-writedata") {
      write_data = true;
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
  
  Group* all = new Group();
  // the value of colordata will be checked later and the
  // program will abort if the value is too large.
  GridSpheresDpy* display = new GridSpheresDpy(colordata-1, dpy_config);
  if (radius_index >= 0)
    display->set_radius_index(radius_index);
  SelectableGroup* alltime = new SelectableGroup(1/rate);
 
  RegularColorMap *cmap = 0;
  if (cmap_file)
    cmap = new RegularColorMap(cmap_file);
  else {
    int cmap_type_index = RegularColorMap::parseType(cmap_type);
    cmap = new RegularColorMap(cmap_type_index);
  }

  try {
    DataArchive* da = new DataArchive(filebase);

    cerr << "Done parsing dataArchive\n";
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;
    da->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    cout << "There are " << vars.size() << " variables:\n";
    for(size_t vi = 0; vi < vars.size(); ++vi) {
      if (types[vi]->getType() == Uintah::TypeDescription::ParticleVariable)
        cout << "var["<<vi<<"] = "<<vars[vi]<<" (type="<<types[vi]->getName()<<")\n";
    }
    
    vector<int> index;
    vector<double> times;
    da->queryTimesteps(index, times);
    ASSERTEQ(index.size(), times.size());
    // This will make sure that when we cast to an int, we don't burn
    // ourselves.
    ASSERTL3(times.size() < INT_MAX);
    cout << "There are " << index.size() << " timesteps:\n";
    
    //------------------------------
    // figure out the lower and upper bounds on the timesteps
    if (time_step_lower <= -1)
      time_step_lower = 0;
    else if (time_step_lower >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      abort();
    }
    if (time_step_upper <= -1)
      time_step_upper = (int)(times.size()-1);
    else if (time_step_upper >= times.size()) {
      cerr << "timesteplow must be between 0 and " << times.size()-1 << endl;
      abort();
    }

    Semaphore* thread_sema = scinew Semaphore("rtrt::uintahparticle semaphore",
                                              rtrt::Min(nworkers,5));
    Semaphore* prepro_sema = scinew Semaphore("rtrt::uintahparticle preprocess semaphore", rtrt::Min(nworkers,8));
    Mutex *amutex = scinew Mutex("rtrt::Append spheres mutex");
    //------------------------------
    // start the data extraction
    
    // data structure for all the spheres
    rtrt::Array1<SphereData> sphere_data;
    
    // for all timesteps
    for(int t=time_step_lower;t<=time_step_upper;t+=time_step_inc){
      //      AuditDefaultAllocator();    
      double time = times[t];
      cerr << "Started timestep t["<<t<<"] = "<<time<<"\n";

      sphere_data.remove_all();
      int patch_count = non_empty_patches;
      
      GridP grid = da->queryGrid(time);
      if(do_verbose)
        cout << "time = " << time << "\n";
      cerr << "Creating new timeblock.\n";
      
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
          thread_sema->down();
          Thread *thrd = scinew Thread(scinew SphereMaker(amutex,
                                                          thread_sema,
                                                          *iter,
                                                          da,
                                                          &sphere_data,
                                                          time,
                                                          &vars,
                                                          &var_include,
                                                          &types,
                                                          do_PTvar_all,
                                                          do_patch,
                                                          do_material,
                                                          do_verbose,
                                                          compute_radius,
                                                          radius,
                                                          radius_factor),
                                       "rtrt::SphereMaker");
          thrd->detach();
              
          if (patch_count > 0) patch_count--;
          if (patch_count == 0) {
            cerr << "Only processing partial number of patches\n";
            break;
          }
          AuditDefaultAllocator();
          if (debug) cerr << "Finished patch\n";
        } // end for(patch)
        AuditDefaultAllocator();          
      if (debug) cerr << "Finished level\n";
      } // end for(level)
      AuditDefaultAllocator();    
      //Material* matl0=new Phong( Color(.2,.2,.2), Color(.3,.3,.3), 10, .5);
      //timeblock2->add(new Sphere(matl0,::Point(t,t,t),1));
      //alltime->add(timeblock2);
      thread_sema->down(rtrt::Min(nworkers,5));
      cout << "Adding timestep.\n";
      GridSpheres* obj = create_GridSpheres(sphere_data,colordata,
                                            gridcellsize,griddepth, cmap,
                                            t);
      thread_sema->up(rtrt::Min(nworkers,5));
      if (total_allocated != total_freed) {
        cerr << "total_allocated != total_freed\n";
        cerr << "total_allocated = "<<total_allocated
             <<", total_freed = "<<total_freed<<endl;
      }
      display->attach(obj);
      alltime->add((Object*)obj);
      prepro_sema->down();
      Thread *thrd = scinew Thread(scinew Preprocessor(obj,prepro_sema),
                                   "rtrt::uintahparticle:Preprocessor Thread");
      thrd->detach();
      //      cerr << "Finished timestep t["<<t<<"]\n";
    } // end timestep
    all->add(alltime);
    if( thread_sema ) delete thread_sema;
    prepro_sema->down(rtrt::Min(nworkers,8));
    if (prepro_sema) delete prepro_sema;
    if (amutex) delete amutex;
    if (da) delete da;
    AuditDefaultAllocator();
    if (debug) cerr << "Finished adding all timesteps.\n";
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
  
  rtrt::Plane groundplane (rtrt::Point(-500, 300, 0), rtrt::Vector(7, -3, 2));
  Camera cam(rtrt::Point(0,0,400),rtrt::Point(0,0,0),rtrt::Vector(0,1,0),60.0);
  Color bgcolor(1.,1.,1.);
  double ambient_scale=0.7;
  Color cup(0.9, 0.7, 0.3);
  Color cdown(0.0, 0.0, 0.2);

  Scene* scene=new Scene(all, cam,
                         bgcolor, cdown, cup, groundplane, 
                         ambient_scale);

  // Add all the lights.
  rtrt::BBox bounds;
  all->compute_bounds(bounds, 0);
  // Take the corners and then extend one of them
  Point light_loc = bounds.max() + (bounds.max()-bounds.min())*0.2;
  
  Light *light = new Light(light_loc, Color(.8, .8, .8),
                           (bounds.max()-bounds.min()).length()*0.01);
  light->name_ = "Main Light";
  scene->add_light(light);

  // Add all the displays.
  scene->attach_display(display);
  display->setName("Particle Vis");
  (new Thread(display, "GridSpheres display thread\n"))->detach();

  // Add objects of interest
  scene->addGuiObject("Timesteps", alltime);

  scene->select_shadow_mode( No_Shadows );
  return scene;
}















