
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>

main(int argc, char* argv[])
{
  if(argc < 8 || argc > 11){
    cerr << "Usage: " << argv[0] << " file id mixture Th Tc viscosity thermalConductivity [pressure g L]\n";
    exit(1);
  }
  string fname = argv[1];
  string id = argv[2];
  string mix = argv[3];
  double Th = atof(argv[4]);
  double Tc = atof(argv[5]);
  double Tmid = (Th+Tc)/2;

  double v = atof(argv[6]);
  double thermalConductivity = atof(argv[7]);
  
  double pressure = 101325;
  if(argc > 8)
    pressure = atof(argv[8]);
  double g = 9.81;
  if(argc > 9)
    g = atof(argv[9]);
  double L = 1;
  if(argc > 10)
    L = atof(argv[10]);
    
  try {
    IdealGasMix* gas = new IdealGasMix(fname, id);
    int nsp = gas->nSpecies();
    int nel = gas->nElements();
    cerr.precision(17);
    cerr << "Using ideal gas " << id << "(from " << fname << ") with " << nel << " elements and " << nsp << " species\n";
    gas->setState_TPY(Tmid, pressure, mix);

    double B = gas->thermalExpansionCoeff();
    double spvol = 1./gas->density();
    double cp = gas->cp_mass();
    double cv = gas->cv_mass();
    double a = thermalConductivity/cp * spvol;

    gas->setState_TPY(Tc, pressure, mix);
    double density_c = gas->density();
    gas->setState_TPY(Tmid, pressure, mix);
    double density_mid = gas->density();
    gas->setState_TPY(Th, pressure, mix);
    double density_h = gas->density();
    double Ra = (g*B*(Th-Tc)*L*L*L)/(v*a);
    cerr.precision(17);
    cerr << "mixture = " << mix << '\n';
    cerr << "pressure = " << pressure << '\n';
    cerr << "specific volume = " << spvol << '\n';
    cerr << "cp = " << cp << '\n';
    cerr << "cv = " << cv << '\n';
    cerr << "gamma = " << cp/cv << '\n';
    cerr << "g = " << g << '\n';
    cerr << "B = " << B << '\n';
    cerr << "Th = " << Th << '\n';
    cerr << "Tmid = " << Tmid << '\n';
    cerr << "Tc = " << Tc << '\n';
    cerr << "L = " << L << '\n';
    cerr << "v = " << v << '\n';
    cerr << "a = " << a << '\n';
    cerr << "density @ Tc = " << density_c << '\n';
    cerr << "density @ Tmid = " << density_mid << '\n';
    cerr << "density @ Th = " << density_h << '\n';
    double speedSound = sqrt(cp/cv*pressure/density_mid);
    cerr << "speed of sound = " << speedSound << '\n';
    cerr << "Ra = " << Ra << '\n';
  } catch (CanteraError) {
    showErrors(cerr);
    exit(1);
  }
}
