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

#ifndef KURT_HARVARDVIS_H
#define KURT_HARVARDVIS_H
/*
 * HarvardVis.cc
 *
 * Visualize Harvard Data
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Containers/Array1.h>

#include <string>

namespace Kurt {

using SCIRun::Module;
using SCIRun::FieldOPort;
using SCIRun::GuiInt;
using SCIRun::GuiDouble;
using SCIRun::GuiString;
using SCIRun::Array1;

class Annuli {
public:
  Annuli():
    Radius(0), KepVel(0), MassInZone(0), MassLost(0), MassLostRate(0)
  {}
  double Radius; // heliocentric distance in AU from the central star
  double KepVel; // the velocity of a circular orbit at that distance

  double MassInZone; // total mass in planetesimals in annulus I
  double MassLost; // amount of mass lost to particles smaller than
                   // the smallest bin
  double MassLostRate; // rate of mass lost

  int nbins; // number of mass bins
  
  void print();
};

class MassBin {
public:
  MassBin(): n(0), ncum(0), ecc(0), h_scl(0), Mtot(0), Mi(0) {}
  
  double n; // number of particles in the mass bin
  double ncum; // cumulative number of particles starting from the most
               // massive bin
  double ecc; // orbital eccentricity of particles in the mass bin
  double h_scl; // scale height above the disk midplane
                // - a proxy for the inclination
  double Mtot; // total mass in the mass bin in grams
  double Mi; // average mass of particles in a mass bin in grams

  void print();
};

class Disk {
public:
  Disk(): num_annuli(0), mass_bins(0), ring_data(0) {}
  Disk(int num_annuli):
    num_annuli(num_annuli)
  {
    mass_bins.resize(num_annuli);
    ring_data.resize(num_annuli);
  }
  ~Disk() {}
  
  Array1<Array1<MassBin> > mass_bins;
  Array1<Annuli> ring_data;
  double DelTime;
  int num_annuli;

  // All data must be positive!!

  
  void minmax_KepVel(double &min_out, double &max_out) const;
  void minmax_MassInZone(double &min_out, double &max_out) const;
  void minmax_MassLost(double &min_out, double &max_out) const;
  void minmax_MassLostRate(double &min_out, double &max_out) const;

  void minmax_MassBin_n(double &min_out, double &max_out) const;
  void minmax_MassBin_ncum(double &min_out, double &max_out) const;
  void minmax_MassBin_ecc(double &min_out, double &max_out) const;
  void minmax_MassBin_h_scl(double &min_out, double &max_out) const;
  void minmax_MassBin_Mtot(double &min_out, double &max_out) const;
  void minmax_MassBin_Mi(double &min_out, double &max_out) const;
};

class HarvardVis : public Module {

public:
  HarvardVis(SCIRun::GuiContext* ctx);

  virtual ~HarvardVis();
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  
  FieldOPort* oport_;

  std::string current_filename;
  bool reread_datafile;
  
  GuiInt which_I_var_;
  GuiInt num_timesteps_, which_timestep_;
  GuiString file_name_;

  Array1<Disk> time_data;

  bool read_data();
  void gen_geom(FieldOPort *curve, int time_index);

  double get_mass_bin_value(const MassBin& mb, int which);
  double get_annuli_value(const Annuli& ring, int which);
  
};

} // End namespace Kurt

#endif // KURT_HARVARDVIS_H
