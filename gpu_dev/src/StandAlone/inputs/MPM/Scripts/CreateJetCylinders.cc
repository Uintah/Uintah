#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;


// define INSERT_P to generate a cylinder distribution that is compatible with
// the insert particles feature

// define FULL_LENGTH to generate the traditional long string of cylinders

//#define FULL_LENGTH
#undef FULL_LENGTH
#define INSERT_P
//#undef INSERT_P




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

  // Density fits
  double ae = -27739.2;  // Tail region of jet
  double be = 8248.14;   // Tail region of jet
  double a = -6200.44;   // Tip region of jet
  double b = 5466.92;    // Tip region of jet

  // Radius fits
  double c = -0.0105897;  // Tail region of jet
  double d =  0.0029218;  // Tail region of jet
  double q = -0.00253423; // Tip region of jet
  double r =  0.001756;   // Tip region of jet
  double rho_W = 17600.0;
  double v_tip=7830.0;
  double L0=0.30;
  double delT=0.25e-6;
  double total_mass=0.;

  double x=0.12;          // Position at which to monitor densities

  dest << "<!--\n" << endl;
  dest << "x = " << x << endl;
  dest << "a = " << a << endl;
  dest << "b = " << b << endl;
  dest << "c = " << c << endl;
  dest << "d = " << d << endl;
  dest << "q = " << q << endl;
  dest << "r = " << r << endl;
  dest << "rho_W = " << rho_W << endl;
  dest << "delT = " << delT << endl;
  dest << "-->\n" << endl;

  double T0=(x-L0)/v_tip; // Time at which tip is at x
  double t=T0; 
  double X=100.;  // Initialize ref position to large value to get started
  int n=0;
  while(X>0.05){  // Continue until the tail of the reference jet is reached
    double F=1.0+v_tip*(t/L0);
    double rho;
    X=x/F;
    double rad = 0.;
    if(X <= .30 && X > .13){
      rad = q*X + r;
      rho=(a*(x/F) + b)/F;
    }
    else if(X <=.13){
      rad = c*X + d;
      rho=(ae*(x/F) + be)/F;
    }
    else{
     cout << "SHOULDN'T GET HERE!  X = " << X << endl;
    }

    double vel = v_tip*X/L0;
    // L_int = Length of this element of the jet
    double L_int=vel*delT; 
    // L_seg = Length of the segment representing this element of the jet
    double L_seg=L_int*(rho/rho_W);

    total_mass+=(M_PI*rad*rad)*L_seg*rho_W;
    // Make the radius of the cylinder slightly smaller than the radius
    // from the curve fit.  This is done for a few reasons:
    // 1.  Accounts for a non-flat density profile.
    // 2.  Reduces interference with the tunnel walls, a problem that wouldn't
    //     occur in a particulated jet
    double volume = (M_PI*rad*rad)*L_seg;
    if(X<0.25){
      rad*=.9;
      L_seg = volume/(M_PI*rad*rad);
    }
    double elapT = t-T0;
    double init_pos = vel*(-elapT) - 0.001;
    dest_IP << t-T0 << " " << n << " " << " 0.0  0.01 0.0  0.0 " << vel << " 0.0\n";
    // adjust delT as to keep cylinders from shrinking too much
    if(X>0.25){
      delT=delT*pow(v_tip/vel,.05);
    } else if(X>.20){
      delT=delT*pow(v_tip/vel,.025);
    } else {
      delT=delT*pow(v_tip/vel,.010);
    } 
    t+=delT;

    // Write cylinder to the cylinders.xml file
    dest << "  <geom_object>\n";
    dest << "    <cylinder label = \"" << n << "\">\n";
#ifdef FULL_LENGTH 
    dest << "       <top>[0.0, " << init_pos << ", 0.0]</top>\n";
    dest << "       <bottom>[0.0, " << init_pos-L_seg << ", 0.0]</bottom>\n";
#endif
#ifdef INSERT_P
    dest << "       <top>[0.0, " << -0.01 << ", 0.0]</top>\n";
    dest << "       <bottom>[0.0, " << -0.01 - L_seg << ", 0.0]</bottom>\n";
#endif
    dest << "       <radius>" << rad << "</radius>\n";
    dest << "    </cylinder>\n";
    dest << "    <res>[4,4,1]</res>\n";
#ifdef FULL_LENGTH 
    dest << "    <velocity>[0.0, " << vel << ", 0.0]</velocity>\n";
#endif
#ifdef INSERT_P
    dest << "    <velocity>[0.0, 0.0, 0.0]</velocity>\n";
#endif
    dest << "    <temperature>294.0</temperature>\n";
    dest << "    <color>" << n++ << "</color>\n";
    dest << "  </geom_object>\n\n";
  }
  cout << "Total Mass = " << total_mass << endl;

  dest << "</Uintah_Include>" << endl;
}
