/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 * HarvardVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Kurt/Dataflow/Modules/Visualization/HarvardVis.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Datatypes/CurveField.h>
#include <Core/Datatypes/CurveMesh.h>

#include <iostream>
#include <fstream>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <sstream>
#include <string>
#include <values.h>
using namespace std;
using namespace SCIRun;
using namespace Kurt;

void Annuli::print() {
  cout << "nbins = "<<nbins;
  cout << ", Radius = "<<Radius;
  cout << ", KepVel = "<<KepVel<<endl;
  cout << "MassInZone = "<<MassInZone;
  cout << ", MassLost = "<<MassLost;
  cout << ", MassLostRate = "<<MassLostRate<<endl;
}

void MassBin::print() {
  cout <<n<<" "<< ncum <<" "<< ecc <<" "<< h_scl <<" "<< Mtot <<" "<< Mi << endl;
}

void Disk::minmax_KepVel(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < ring_data.size(); i++) {
    if (ring_data[i].KepVel < min)
      min = ring_data[i].KepVel;
    if (ring_data[i].KepVel > max)
      max = ring_data[i].KepVel;
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassInZone(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < ring_data.size(); i++) {
    if (ring_data[i].MassInZone < min)
      min = ring_data[i].MassInZone;
    if (ring_data[i].MassInZone > max)
      max = ring_data[i].MassInZone;
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassLost(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < ring_data.size(); i++) {
    if (ring_data[i].MassLost < min)
      min = ring_data[i].MassLost;
    if (ring_data[i].MassLost > max)
      max = ring_data[i].MassLost;
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassLostRate(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < ring_data.size(); i++) {
    if (ring_data[i].MassLostRate < min)
      min = ring_data[i].MassLostRate;
    if (ring_data[i].MassLostRate > max)
      max = ring_data[i].MassLostRate;
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}

void Disk::minmax_MassBin_n(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < mass_bins.size(); i++) {
    //      cout << "mass_bins["<<i<<"].size = "<<mass_bins[i]->size()<<endl;
    for(int j = 0; j < mass_bins[i].size(); j++) {
      double val = mass_bins[i][j].n;
      //	cout << "val("<<val<<")";
      if (val < min)
	min = val;
      if (val > max)
	max = val;
    }
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassBin_ncum(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < mass_bins.size(); i++) {
    for(int j = 0; j < mass_bins[i].size(); j++) {
      double val = mass_bins[i][j].ncum;
      if (val < min)
	min = val;
      if (val > max)
	max = val;
    }
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassBin_ecc(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < mass_bins.size(); i++) {
    for(int j = 0; j < mass_bins[i].size(); j++) {
      double val = mass_bins[i][j].ecc;
      if (val < min)
	min = val;
      if (val > max)
	max = val;
    }
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassBin_h_scl(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < mass_bins.size(); i++) {
    for(int j = 0; j < mass_bins[i].size(); j++) {
      double val = mass_bins[i][j].h_scl;
      if (val < min)
	min = val;
      if (val > max)
	max = val;
    }
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassBin_Mtot(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < mass_bins.size(); i++) {
    for(int j = 0; j < mass_bins[i].size(); j++) {
      double val = mass_bins[i][j].Mtot;
      if (val < min)
	min = val;
      if (val > max)
	max = val;
    }
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  
void Disk::minmax_MassBin_Mi(double &min_out, double &max_out) const {
  // Loop over all the ring_data and determine the min and max
  double min = DBL_MAX;
  double max = -DBL_MAX;
  for(int i = 0; i < mass_bins.size(); i++) {
    for(int j = 0; j < mass_bins[i].size(); j++) {
      double val = mass_bins[i][j].Mi;
      if (val < min)
	min = val;
      if (val > max)
	max = val;
    }
  }
  if (min < min_out)
    min_out = min;
  if (max > max_out)
    max_out = max;
}
  

static int parseI(ifstream &infile, Disk &disk, int index) {
  //  cout << "I index = "<<index<<endl;
  Annuli ring_data;
  
  infile >> ring_data.nbins >> ring_data.Radius >> ring_data.KepVel;
  
  // Allocate the Mass Bins
  disk.mass_bins[index].resize(ring_data.nbins);
  
  int j;
  do {
    infile >> j;
    MassBin mass;
    infile >> mass.n >> mass.ncum >> mass.ecc >> mass.h_scl >> mass.Mtot
	   >> mass.Mi;
    disk.mass_bins[index][j] = mass;
    //    disk.mass_bins[index][j].print();
  } while (j != ring_data.nbins-1);

  infile >> ring_data.MassInZone >> ring_data.MassLost
	 >> ring_data.MassLostRate;

  disk.ring_data[index] = ring_data;
  //  disk.ring_data[index].print();
  return 0;
}

static int parseDisk(ifstream &infile, Disk &disk) {
  infile >> disk.DelTime;
  if (!infile)
    // There is no more stuff to parse, so quit
    return 1;
  cout << "Reading timestep: "<<disk.DelTime<<endl;
  
  for(int i = 0; i < disk.num_annuli; i++) {
    int I_index;
    infile >> I_index;
    parseI(infile, disk, I_index);
    //    cout << "parseDisk:disk.mass_bins["<<I_index<<"].size = "<<disk.mass_bins[I_index]->size()<<endl;
  }

  return 0;
}

static int parsefile(const char* filename, Array1<Disk> &data) {
  char me[] = "parsefile";
  cout << "Attempting to parse "<<filename<<"\n";
  
  // Open the file
  ifstream infile(filename);
  if(!infile){
    fprintf(stderr, "%s: Error opening file: %s\n", me, filename);
    return 1;
  }
  // Parse the name of the file
  string filenameP(""), token("");

  infile >> filenameP;
  cout << "File identified itself as: "<<filenameP<<endl;

  int zones;
  infile >> zones;
  int bins;
  infile >> bins;

  cout << "zones = "<<zones<<", bins = "<<bins<<endl;

  // Need to do this for each timestep
  do {  
    // Parse out a disk
    Disk disk(zones);
    if (!parseDisk(infile, disk))
      data.add(disk);
  } while(infile);
  
  return 0;
}

DECLARE_MAKER(HarvardVis)

HarvardVis::HarvardVis(GuiContext* ctx)
  : Module("HarvardVis", ctx,  Filter, "Visualization", "Kurt"),
    current_filename(""), reread_datafile(true),
    which_I_var_(ctx->subVar("which_I_var_")),
    num_timesteps_(ctx->subVar("num_timesteps_")),
    which_timestep_(ctx->subVar("which_timestep_")),
    file_name_(ctx->subVar("file_name_")),
    time_data(0)
{
}

HarvardVis::~HarvardVis()
{

}

bool HarvardVis::read_data() {
  // Check to see if we need to reread the data
  string newfilename = file_name_.get();
  if (newfilename != current_filename)
    reread_datafile = true;
  // If we need to reread the data, empty the geometry
  if (reread_datafile) {
    time_data.resize(0);
    // Parse the data
    parsefile(file_name_.get().c_str(),time_data);
    reread_datafile = false;
    current_filename = newfilename;
  }

  // Update the gui state relating to the number of time steps
  num_timesteps_.set(time_data.size()-1);
  if (which_timestep_.get() > num_timesteps_.get()) {
    which_timestep_.set(num_timesteps_.get());
  }
  gui->execute(id + " update_slider");
  
  return true;
}

double HarvardVis::get_mass_bin_value(const MassBin& mb, int which) {
  switch (which) {
  case 0:
    return mb.n;
  case 1:
    return mb.ncum;
  case 2:
    return mb.ecc;
  case 3:
    return mb.h_scl;
  case 4:
    return mb.Mtot;
  case 5:
    return mb.Mi;
  default:
    error("Unrecognized option for get_mass_bin_value");
    return 0;
  }
}

double HarvardVis::get_annuli_value(const Annuli& ring, int which) {
  switch (which) {
  case 10:
    return ring.Radius;
  case 11:
    return ring.KepVel;
  case 12:
    return ring.MassInZone;
  case 13:
    return ring.MassLost;
  case 14:
    return ring.MassLostRate;
  default:
    error("Unrecognized option for get_annuli_value");
    return 0;
  }
}

void HarvardVis::gen_geom(FieldOPort *curve, int time_index) {
  if (time_index < 0) {
    return;
  }
  
  // Create the mesh and field
  CurveMeshHandle mesh = scinew CurveMesh();
  CurveField<double> *field = scinew CurveField<double>(mesh, Field::NODE);

  // Now iterate trough each ring and add the curve.
  CurveMesh::Node::index_type start, n1, n2;
  Point node_location;
  double value = 0;
  bool use_ring_value = which_I_var_.get() >= 10;

  Disk *disk = &(time_data[time_index]);
  for(int ring_index = 0; ring_index < disk->ring_data.size(); ring_index++) {
    //    cout << "ring["<<ring_index<<"].Radius = "<<disk->ring_data[ring_index].Radius;
    // Figure out how many mass bins go around the ring
    int num_massbins = disk->mass_bins[ring_index].size();
    //    cout << ", num_massbins = "<<num_massbins<<endl;
    double radius = disk->ring_data[ring_index].Radius;
    double height = 0; // this value could be from another field

    /////////////////////////////////////////////////////
    // Compute all the node locations for this ring

    // Do the first node, so that we can use the index to complete the loop
    node_location = Point(radius, 0, height);
    start = n1 = field->get_typed_mesh()->add_node(node_location);
    field->resize_fdata();
    if (use_ring_value)
      value = get_annuli_value(disk->ring_data[ring_index],
			       which_I_var_.get());
    else
      value = get_mass_bin_value(disk->mass_bins[ring_index][0],
				       which_I_var_.get());
      //    value = disk->mass_bins[ring_index][0].Mi;
    //value = 0;
    field->set_value(value, start);
    for(int mass_index = 1; mass_index < num_massbins; mass_index++) {
      // Add the node
      double x, y, theta;
      theta = 2 * M_PI * mass_index / num_massbins;
      x = radius * cos(theta);
      y = radius * sin(theta);
      node_location = Point(x, y, height);
      n2 = field->get_typed_mesh()->add_node(node_location);
      field->resize_fdata();
      if (!use_ring_value)
	value = get_mass_bin_value(disk->mass_bins[ring_index][mass_index],
				   which_I_var_.get());
      //value = mass_index;
      field->set_value(value, n2);
      field->get_typed_mesh()->add_edge(n1, n2);
      n1 = n2;
    }
    // Add an edge to close the circle
    field->get_typed_mesh()->add_edge(n1, start);
  }

  field->freeze();
  
  curve->send(field);
}

void HarvardVis::execute(void)
{
  oport_ = (FieldOPort *)get_oport("Curve Field");

  if (!oport_) {
    error("Unable to initialize "+name+"'s Curve Field\n");
    return;
  }

  // Read in the geometry if we need to.
  reset_vars();
  if (read_data())
    gen_geom(oport_, which_timestep_.get());
}




