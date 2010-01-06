#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

/*
Create a series of cylinders with the density of pure tungsten at room
temperature that models a shape charge jet as it passes point "x".  The
cylinders that comprise the jet are stacked atop each other initially.
The InsertParticles.dat file that is also created describes the translation
of those cylinders to the front of the target at the appropriate time.

To Compile:

g++ -o CreateJetCylinders CreateJetCylinders.cc

*/

int main()
{
  string outfile_name = "cylinders.xml";
  ofstream dest(outfile_name.c_str());
  if(!dest){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  outfile_name = "InsertParticles.dat";
  ofstream dest_IP(outfile_name.c_str());
  if(!dest_IP){
    cerr << "File " << outfile_name << " can't be opened." << endl;
  }

  dest << "<?xml version='1.0' encoding='ISO-8859-1' ?>" << endl;
  dest << "<Uintah_Include>" << endl;

  double A = 3.937;
  double B = 4.724;
  double rad_ref = .004318;
  double rho_ref = 16000.0;
  double rho_W = 17600.0;
  double v_tip=7830.0;
  double c=1.7;
  double L0=0.30;
  double delT=0.5e-6;
  double total_mass=0.;

  double x=0.10;     // Position at which to monitor densities
  double T0=-25.543e-6; // Time at which tip is at x=0.10 m
  double t=T0;          // Time at which tip is at x=0.10 m
  double temp_pos = -0.001;
  double X=100.;  // Initialize ref position to large value to get started
  int n=0;
  while(X>0.05){
    double rho=(L0*rho_ref)/(L0+v_tip*t+A*x);
    double F=1.0+v_tip*(t/L0);
    X=x/F;
    double rad = rad_ref/(1.0+B*X/L0);
    double vel = v_tip*X/L0-c;
    double L_int=vel*delT;
    double L_seg=L_int*(rho/rho_W);
    cout << t << " " << rho << " " << rad << " " << vel << " " << L_int << " " << L_seg << endl;
    total_mass+=(M_PI*rad*rad)*L_seg*rho_W;
    delT=delT*pow(v_tip/vel,.05);
//    dest_IP << t-T0 << " " << n << " " << " 0.0 " << -temp_pos << " 0.0 0.0 " << vel << "0.0\n";
    dest_IP << t-T0 << " " << n << " " << " 0.0 " <<  0.02 << " 0.0 0.0 " << vel << "0.0\n";
    t+=delT;
    dest << "  <geom_object>\n";
    dest << "    <cylinder label = \"" << n << "\">\n";
//    dest << "       <top>[0.0, " << temp_pos << ", 0.0]</top>\n";
//    dest << "       <bottom>[0.0, " << temp_pos-L_seg << ", 0.0]</bottom>\n";
    dest << "       <top>[0.0, " << -0.02 << ", 0.0]</top>\n";
    dest << "       <bottom>[0.0, " << -0.02 - L_seg << ", 0.0]</bottom>\n";
    dest << "       <radius>" << rad << "</radius>\n";
    dest << "    </cylinder>\n";
    dest << "    <res>[2,2,1]</res>\n";
    dest << "    <velocity>[0.0, 0.0, 0.0]</velocity>\n";
    dest << "    <temperature>294.0</temperature>\n";
    dest << "    <color>" << n++ << "</color>\n";
    dest << "  </geom_object>\n\n";
    temp_pos -= L_int;
  }
  cout << "Total Mass = " << total_mass << endl;

  dest << "</Uintah_Include>" << endl;
}
