#include <Core/ProblemSpec/ProblemSpec.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <CCA/Components/Arches/ChemMix/ChemHelper.h>
#include <CCA/Components/Arches/SourceTerms/MoMICSoot.h>

using namespace std;
using namespace Uintah;

// Declare constants and functions used in this model
  
const double rhos = 1850.0;                   ///< soot density       (kg/m3)
const double kB   = 1.3806485279e-23;         ///< Boltzmann constant (kg*m2/s2*K)
const double Na   = 6.0221413e26;             ///< Avogadro's number  (#/kmole)
const double Rgas = kB*Na;                    ///< Ideal gas constant (J/K*kmole)
const double mC   = 12.01/Na;                 ///< Carbon MW          (kg)
const double m0   = 400.0/Na;                 ///< Base particle MW   (kg)
const double Visc = 5.5e-5;                   ///< Gas viscosity approximation (kg/m*s)
const double da   = 1.395e-10*pow(3.0,0.5);   ///< Diameter of a single aromatic ring (m)
const double Cd   = 1.9125;                   ///< Empirical fitted proportionality constant
const double Ch   = da*pow(2.0/(3.0*mC),0.5); ///< Precursor diameter constant (m/kg^0.5)
const double Cs   = pow(6.0/(M_PI*rhos),1.0/3.0); ///< Soot particle diameter constant (m/kg^0.33)

double
Combination(int r,
    	      int k)
{
    /// Function computes statistical combination of r and k
    double rfact       = 1.0;
    double kfact       = 1.0;
    double rminkfact   = 1.0;
    for (int i=0;i<r;i++){
        rfact = rfact*(r-i);
    }
    for (int i=0;i<k;i++){
        kfact = kfact*(k-i);
    }
    for (int i=0;i<r-k;i++){
        rminkfact = rminkfact*(r-k-i);
    }
    return rfact/(kfact*rminkfact);
}

double
LagrangianInterpolation(double xi,
                        std::vector<int> x,
                        std::vector<double> y)
{
    /// Function calculates the desired fractional value by Lagrange interpolation
    //
    // xi is the x coordinate of the desired value
    // x and y are the coordinates of known data
    ///

    double yi=0.0;

    for (unsigned int i=0;i<y.size();i++){
        double l = 1.0;
        for (unsigned int j=0;j<x.size();j++){
            if (j!=i){
                l = l*(xi-x[j])/(x[i]-x[j]);
            }
        }
        yi = yi+y[i]*l;
    }

    return yi;
}

double
MoMIC(double p,
      std::vector<double> Moments)
{
    /// Function computes desired fractional moment (p) by Lagragnian interpolation among whole-order moments
    std::vector<double> log_Moments(Moments.size());
    for (unsigned int i=0;i<Moments.size();i++) {
        log_Moments[i] = log10(Moments[i]/Moments[0]);
    }
    std::vector<int> x(Moments.size());
    for (unsigned int i=0;i<Moments.size();i++){
        x[i] = i;
    }
    double log_p = LagrangianInterpolation(p,x,log_Moments);
    return pow(10.0,log_p)*Moments[0];
}

double
fm_coag_grid(double l,
             int    x,
             int    y,
             std::vector<double> Moments)
{
    /// Function calculates grid function for coagulation in the free-molecular regime
    //
    // l power for the additive masses
    // x and y are grid values
    // Moments is a matrix of the previous moments
    ///
    
    std::vector<double> L(4);
    L[0] = log10(MoMIC(double(x-1.0/2.0),Moments)*MoMIC(double(y+1.0/6.0),Moments)+2.0*MoMIC(double(x-1.0/6.0),Moments)*MoMIC(double(y-1.0/6.0),Moments)+MoMIC(double(x+1.0/6.0),Moments)*MoMIC(double(y-1.0/2.0),Moments));            
    L[1] = log10(MoMIC(double(x-1.0/2.0),Moments)*MoMIC(double(y+7.0/6.0),Moments)+2.0*MoMIC(double(x-1.0/6.0),Moments)*MoMIC(double(y+5.0/6.0),Moments)+MoMIC(double(x+1.0/6.0),Moments)*MoMIC(double(y+1.0/2.0),Moments)+MoMIC(double(x+1.0/2.0),Moments)*MoMIC(double(y+1.0/6.0),Moments)+2.0*MoMIC(double(x+5.0/6.0),Moments)*MoMIC(double(y-1.0/6.0),Moments)+MoMIC(double(x+7.0/6.0),Moments)*MoMIC(double(y-1.0/2.0),Moments));
    L[2] = log10(MoMIC(double(x-1.0/2.0),Moments)*MoMIC(double(y+13.0/6.0),Moments)+2.0*MoMIC(double(x-1.0/6.0),Moments)*MoMIC(double(y+11.0/6.0),Moments)+MoMIC(double(x+1.0/6.0),Moments)*MoMIC(double(y+3.0/2.0),Moments)+2.0*MoMIC(double(x+1.0/2.0),Moments)*MoMIC(double(y+7.0/6.0),Moments)+4.0*MoMIC(double(x+5.0/6.0),Moments)*MoMIC(double(y+5.0/6.0),Moments)+2.0*MoMIC(double(x+7.0/6.0),Moments)*MoMIC(double(y+1.0/2.0),Moments)+MoMIC(double(x+3.0/2.0),Moments)*MoMIC(double(y+1.0/6.0),Moments)+2.0*MoMIC(double(x+11.0/6.0),Moments)*MoMIC(double(y-1.0/6.0),Moments)+ MoMIC(double(x+13.0/6.0),Moments)*MoMIC(double(y-1.0/2.0),Moments));
    L[3] = log10(MoMIC(double(x-1.0/2.0),Moments)*MoMIC(double(y+19.0/6.0),Moments)+2.0*MoMIC(double(x-1.0/6.0),Moments)*MoMIC(double(y+17.0/6.0),Moments)+MoMIC(double(x+1.0/6.0),Moments)*MoMIC(double(y+5.0/2.0),Moments)+3.0*MoMIC(double(x+1.0/2.0),Moments)*MoMIC(double(y+13.0/6.0),Moments)+6.0*MoMIC(double(x+5.0/6.0),Moments)*MoMIC(double(y+11.0/6.0),Moments)+3.0*MoMIC(double(x+7.0/6.0),Moments)*MoMIC(double(y+3.0/2.0),Moments)+3.0*MoMIC(double(x+3.0/2.0),Moments)*MoMIC(double(y+7.0/6.0),Moments)+6.0*MoMIC(double(x+11.0/6.0),Moments)*MoMIC(double(y+5.0/6.0),Moments)+3.0*MoMIC(double(x+13.0/6.0),Moments)*MoMIC(double(y+1.0/2.0),Moments)+MoMIC(double(x+5.0/2.0),Moments)*MoMIC(double(y+1.0/6.0),Moments)+2.0*MoMIC(double(x+17.0/6.0),Moments)*MoMIC(double(y-1.0/6.0),Moments)+MoMIC(double(x+19.0/6.0),Moments)*MoMIC(double(y-1.0/2.0),Moments));
    
    std::vector<int> xmat(4);
    for (int i=0;i<4;i++){
        xmat[i]=i;
    }
    return pow(10,LagrangianInterpolation(l,xmat,L));
}

double
f_grid_surface_reactions(double d,
                         double deltaM,
                         std::vector<double> Moments)
{
    /// Function calculates the grid function for the surface moment surface reactions
    //
    // d is the particles' shape factor
    // deltaM is the change of particle mass due to a single reaction
    // Moments is a vector of the previous resolved moments
    ///

    std::vector<double> L(4);
    L[0] = log10(MoMIC(d,Moments));
    L[1] = log10(MoMIC(1+d,Moments) + deltaM*MoMIC(d,Moments));
    L[2] = log10(MoMIC(2+d,Moments) + 2.0*deltaM*MoMIC(1+d,Moments) + pow(deltaM,2.0)*MoMIC(d,Moments));
    L[3] = log10(MoMIC(3+d,Moments) + 3.0*deltaM*MoMIC(2+d,Moments) + 3.0*pow(deltaM,2.0)*MoMIC(1+d,Moments) + pow(deltaM,3.0)*MoMIC(d,Moments));

    std::vector<int> xmat(4);
    for (int i=0;i<4;i++){
        xmat[i]=i;
    }
    return pow(10,LagrangianInterpolation(d,xmat,L));
}

double
Oxidation(double Temp,
          double PressO2,
          double PressOH)
{
    /// Function calculates a rate of particle oxidation in terms of kg/s*m2 of surface area
    //
    // Temp is the local temperature:               K
    // PressO2 is the local partial pressure of O2: Pa
    // PressOH is the local partial pressure of OH: Pa
    ///
    
    const double AO2 = 7.98E-1;     ///< pre-exponential constant:  kg*K^0.5/Pa*m2*s
    const double EO2 = -1.77E8;     ///< activation energy:         J/kmole
    const double AOH = 1.89E-3;     ///< pre-exponentail constant:  kg*K^0.5/Pa*m2*s
    
    return (AO2*PressO2*exp(EO2/Rgas/abs(Temp))+AOH*PressOH)/pow(abs(Temp),0.5);
}

double
Gasification(double Temp,
             double PressCO2,
             double PressH2O)
{
    /// Function calculates a rate of particle gasification in terms of kg/s*m2 of surface area
    //
    // Temp is the local temperature:                K
    // PressO2 is the local partial pressure of CO2: Pa
    // PressOH is the local partial pressure of H2O: Pa
    ///
    
    const double ACO2 = 3.06E-17;    ///< pre-exponential constant:  kg/Pa^0.5*K2*m2*s
    const double ECO2 = -5.56E6;     ///< activation energy:         J/kmole
    const double AH2O = 6.27E4;      ///< pre-exponentail constant:  kg*K^0.5/Pa^1.21*m2*s
    const double EH2O = -2.95E8;    ///< activation energy:         J/kmole
    
    return ACO2*pow(PressCO2,0.5)*pow(abs(Temp),2.0)*exp(ECO2/Rgas/abs(Temp))+AH2O*pow(abs(Temp),-0.5)*pow(PressH2O,0.13)*exp(EH2O/Rgas/abs(Temp));
}

double
HACA(double Temp,
     double PressC2H2,
     double PressH,
     double PressH2,
     double PressO2,
     double PressOH,
     double PressH2O,
     std::vector<double> Moments)
{
    /// Function calculates a rate of particle surface growth in terms of kg/s*m2 of surface area
    //
    // Temp is the local temperature:                   K
    // PressC2H2, PressH, PressH2, PressO2, PressOH, and PressH2O are the local partial pressures of C2H2, H, O2, OH, and H2O respectivily:                             Pa
    // Moments is a matrix of the previously computed moments
    ///

    double k1  = 4.2e10*exp(-5.4392e7/Rgas/abs(Temp));                  ///< m3/kmole*s
    double kr1 = 3.9e9*exp(-4.6024e7/Rgas/abs(Temp));                  ///< m3/kmole*s
    double k2  = 1.0e7*pow(abs(Temp),0.734)*exp(-5.98312e6/Rgas/abs(Temp)); ///< m3/kmole*s
    double kr2 = 3.68e5*pow(abs(Temp),1.139)*exp(-7.15464e6/Rgas/abs(Temp));///< m3/kmole*s
    double k3  = 2.0e10;                                             ///< m3/kmole*s
    double k4  = 8.0e4*exp(-1.58992e7/Rgas/abs(Temp));                 ///< m3/kmole*s

    double CC2H2 = PressC2H2/Rgas/abs(Temp);   ///< kmole/m3
    double CH    = PressH/Rgas/abs(Temp);      ///< kmole/m3
    double CH2   = PressH2/Rgas/abs(Temp);     ///< kmole/m3
    double COH   = PressOH/Rgas/abs(Temp);     ///< kmole/m3
    double CH2O  = PressH2O/Rgas/abs(Temp);    ///< kmole/m3

    double a = 12.65-0.00563*abs(Temp);
    double b = -1.38+0.00068*abs(Temp);

    double alpha = abs(tanh(a/log10(Moments[1]/Moments[0])+b));
    double chiC  = 2.3e19;         ///< sites/m2
    double mC2H2 = 26.0/Na;       ///< kg/site
    return k4*CC2H2*alpha*chiC*(k1*CH+k2*COH)*mC2H2/(kr1*CH2+kr2*CH2O+k3*CH+k4*CC2H2);
}

double 
SurfaceGrowth(double Temp,
              double PressC2H2,
              double PressH,
              double PressH2,
              double PressO2,
              double PressOH,
              double PressH2O,
              double PressCO2,
              std::vector<double> Moments,
              int r,
              double d)
{
    /// Function computes the change in soot PSD moments based on the surface reactions between gas and particles
    //
    // Temp is the local temperature (K)
    // P_C2H2, P_H, P_H2, P_O2, P_OH, P_H2O, and P_CO2 are the partial pressures of acetylene, mono-hydrogen,
    //         hydrogen, oxygen, hydroxide, water, and carbon dioxide respectively (Pa).
    // PahPsd is a matrix of the previous PAH sectional statistics
    // PahBin is a matrix of the molecular weight of the PAH sections
    // Moments is a matrix of the previously computed moments
    // r is the currently computed moments
    // d is the particle shape factor
    ///

    // Surface reactions do not affect the zeroth moment. Short cut to save on computational time
    if (r==0) {
        return 0;
    }

    // Rate of growth via HACA
    double RateHACA = HACA(Temp,PressC2H2,PressH,PressH2,PressO2,PressOH,PressH2O,Moments);
    double WC2H2 = 0.0;
    for (int k=0;k<r;k++) {
        WC2H2 = WC2H2+Combination(r,k)*pow(2.0*mC,r-k)*MoMIC(k+d,Moments);
    }
    WC2H2 = M_PI*pow(Cs,2.0)*RateHACA/(2.0*mC)*pow(m0,2.0/3.0-d)*WC2H2;

    // Rate of oxidation
    double RateOx = Oxidation(Temp,PressO2,PressOH);
    double WO = 0.0;
    for (int k=0;k<r;k++) {
        WO = WO+Combination(r,k)*pow(mC,r-k)*MoMIC(k+d,Moments);
    }
    WO = M_PI*pow(Cs,2.0)*RateOx/(mC)*pow(m0,2.0/3.0-d)*WO;

    // Rate of gasification
    double RateGa = Gasification(Temp,PressCO2,PressH2O);
    double WG = 0.0;
    for (int k=0;k<r;k++) {
        WG = WG+Combination(r,k)*pow(mC,r-k)*MoMIC(k+d,Moments);
    }
    WG = M_PI*pow(Cs,2.0)*RateGa/(mC)*pow(m0,2.0/3.0-d)*WG;
    
    // Combine it all together
    return WC2H2-WG-WO;
}

double
SootCoagulation(double Temp,
        		    double Press,
		            std::vector<double> Moments,
		            int r,
                double d,
                double time_step)
{
    /// Function computes the change in soot PSD moments based on the coagulation of particles
    // 
    // Temp is the local temperature (K)
    // Press is the local pressure (Pa)
    // Moments is a matrix of the previous moments
    // r is this calculated moment
    ///

    // Coagulation does not affect the first Moment. Shortcut to save on computational time
    
    if (r==1) {
       return 0;
    }
    double temp_time  = time_step;
    double total_time = 0;
    double Ca       = (3.0-3.0*d)+(3.0*d-2.0)*Cd;
    double Dg 		  = pow(6.0*kB*Temp/(Press*M_PI),1.0/3.0);    		  ///< average gas molecular diameter (m)
    double lambdaG 	= kB*Temp/(pow(2.0,0.5)*M_PI*pow(Dg,2.0)*Press);	///< gas mean free path 		    (m)
    double Kc		= 2.0*kB*Temp/(3.0*Visc);
    double Kcprime	= 2.514*lambdaG/(Cs*Ca);
    double Kf = 2.2*pow(Ca*Cs,2.0)*pow(M_PI*kB*Temp/2.0,0.5);
    std::vector<double> temp_Moments(6);
    for(int i=0; i<6;i++) temp_Moments[i] = Moments[i];
    if (r==0 and temp_Moments[0]>1E18) temp_Moments[0] = 1E18;
    while (total_time<time_step){
      double RateC = 0;
      double RateF = 0;
    
      // Continuum Regime
    
      if (r==0) {
        RateC = -1.0*Kc*(pow(temp_Moments[0],2.0)+MoMIC(1.0/3.0,temp_Moments)*MoMIC(-1.0/3.0,temp_Moments)+
    	  		    Kcprime*(MoMIC(-1.0/3.0,temp_Moments)*temp_Moments[0]+MoMIC(-2.0/3.0,temp_Moments)*MoMIC(1.0/3.0,temp_Moments)));
      }
      else {
        for (int k=1;k<r;k++){
    	    RateC = RateC+Combination(r,k)*(2.0*temp_Moments[k]*temp_Moments[r-k]+MoMIC(k+1.0/3.0,temp_Moments)*MoMIC(r-k-1.0/3.0,temp_Moments)+MoMIC(k-1.0/3.0,temp_Moments)*MoMIC(r-k+1.0/3.0,temp_Moments)+Kcprime*(MoMIC(k-1.0/3.0,temp_Moments)*temp_Moments[r-k]+temp_Moments[k]*MoMIC(r-k-1.0/3.0,temp_Moments)+MoMIC(k+1.0/3.0,temp_Moments)*MoMIC(r-k-2.0/3.0,temp_Moments)+MoMIC(k-2.0/3.0,temp_Moments)*MoMIC(r-k+1.0/3.0,temp_Moments)));
    	  }
        RateC = 0.5*Kc*RateC;
      }
    
      // Free-molecular Regime
    
      if (r==0) {
          RateF = -0.5*Kf*fm_coag_grid(0.5,0.0,0.0,temp_Moments);
      }
      else {
          for (int k=1;k<r;k++) {
    	      RateF = RateF+Combination(r,k)*fm_coag_grid(0.5,k,r-k,temp_Moments);
          }
          RateF = 0.5*Kf*RateF;
      }
      double Rate = RateF*RateC/(RateF+RateC);
      
      if (temp_Moments[r]+Rate*temp_time<0){
        temp_time = temp_Moments[r]/abs(Rate)/2;
      } else{
          temp_Moments[r] = temp_Moments[r]+Rate*temp_time;
          total_time = total_time+temp_time;
          temp_time = time_step-total_time;
      }
    }
    return (temp_Moments[r]-Moments[r])/time_step;
}

double
SootSurfaceCoagulation( double Temp,
                        double Press,
                        std::vector<double> Moments,
                        double d)
{
    /// Function calculates the effect of particle coagulation on the surface moment.
    //
    // Temp is the local temperature    (K)
    // Press is the local pressure      (Pa)
    // Moments are the previously resolved soot moments
    // d is the particle shape factor
    ///

    // Compute the Knudsen number for regime determination
    const double Ca = (3.0-3.0*d)+(3.0*d-2.0)*Cd;
    double Dg = pow(6.0*kB*Temp/(Press*M_PI),1.0/3.0);
    //double Dp 	 = Cs*Ca*pow(Moments[1]/Moments[0],1.0/3.0);       ///< average particle diameter		(m)
    double lambdaG = kB*Temp/(pow(2.0,0.5)*M_PI*pow(Dg,2.0)*Press);
    //double Kn = lambdaG/Dp; // knudsen number unused?

    // Continuum Regime
    const double Kc = 2.0*kB*Temp/(3.0*Visc);
    const double Kcprime = 2.514*lambdaG/(Ca*Cs);

    std::vector<double> Term1temp(4);
    Term1temp[0] = log10(pow(Moments[0],2.0)+MoMIC(-1.0/3.0,Moments)+Kcprime*(Moments[0]*MoMIC(-1.0/3.0,Moments)+MoMIC(1.0/3.0,Moments)*MoMIC(-2.0/3.0,Moments)));
    Term1temp[1] = log10(2*Moments[1]*Moments[0]+MoMIC(2.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)+MoMIC(4.0/3.0,Moments)*MoMIC(-1.0/3.0,Moments)+Kcprime*(Moments[0]*MoMIC(2.0/3.0,Moments)+MoMIC(-2.0/3.0,Moments)*MoMIC(4.0/3.0,Moments)+MoMIC(-1.0/3.0,Moments)*Moments[1]+pow(MoMIC(1.0/3.0,Moments),2.0)));
    Term1temp[2] = log10(2*Moments[2]*Moments[0]+2*pow(Moments[1],2.0)+MoMIC(1.0/3.0,Moments)*MoMIC(5.0/3.0,Moments)+2*MoMIC(2.0/3.0,Moments)*MoMIC(4.0/3.0,Moments)+MoMIC(-1.0/3.0,Moments)*MoMIC(7.0/3.0,Moments)+Kcprime*(MoMIC(5.0/3.0,Moments)*Moments[0]+MoMIC(7.0/3.0,Moments)*MoMIC(-2.0/3.0,Moments)+MoMIC(4.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)+Moments[2]*MoMIC(-1.0/3.0,Moments)+2.0*MoMIC(2.0/3.0,Moments)*Moments[1]+2.0*MoMIC(4.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)));
    Term1temp[3] = log10(2*Moments[3]*Moments[0]+6.0*Moments[2]*Moments[1]+MoMIC(8.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)+3.0*MoMIC(5.0/3.0,Moments)*MoMIC(4.0/3.0,Moments)+3.0*MoMIC(2.0/3.0,Moments)*MoMIC(7.0/3.0,Moments)+MoMIC(-1.0/3.0,Moments)*MoMIC(10.0/3.0,Moments)+Kcprime*(MoMIC(8.0/3.0,Moments)*Moments[0]+3.0*MoMIC(5.0/3.0,Moments)*Moments[1]+3.0*Moments[2]*MoMIC(2.0/3.0,Moments)+MoMIC(-1.0/3.0,Moments)*Moments[3]+3.0/2.0*MoMIC(10.0/3.0,Moments)*MoMIC(-2.0/3.0,Moments)+7.0/2.0*MoMIC(7.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)+3.0*pow(MoMIC(4.0/3.0,Moments),2)));

    std::vector<int> x(4);
    for (int i=0;i<4;i++){
        x[i]=i;
    }
    double Term1C = Kc/2.0*pow(10,LagrangianInterpolation(d,x,Term1temp));
    double Term2C = Kc*(2.0*MoMIC(d,Moments)*Moments[0]+MoMIC(d-1.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)+MoMIC(d+1.0/3.0,Moments)*MoMIC(-1.0/3.0,Moments)+Kcprime*(Moments[0]*MoMIC(d-1.0/3.0,Moments)+MoMIC(d+1.0/3.0,Moments)*MoMIC(-2.0/3.0,Moments)+MoMIC(d-2.0/3.0,Moments)*MoMIC(1.0/3.0,Moments)+MoMIC(d,Moments)*MoMIC(-1.0/3.0,Moments)));

    double RateC = Term1C-Term2C;

    // Free-molecular Regime
    const double Kf = 2.2*pow(Ca*Cs*M_PI*kB*Temp/2.0,0.5);

    double Term1F = Kf/2.0*fm_coag_grid(d+1.0/2.0,0,0,Moments);
    double Term2F = Kf*fm_coag_grid(1.0/2.0,d,0,Moments);   
    double RateF = Term1F-Term2F;

    return (RateF*RateC)/(RateF+RateC);
}

//===========================================================================

MoMICSoot::MoMICSoot( std::string src_name, ArchesLabel* field_labels,
                                                    std::vector<std::string> req_label_names, std::string type )
: SourceTermBase(src_name, field_labels->d_materialManager, req_label_names, type), _field_labels(field_labels)
{

  _src_label = VarLabel::create( src_name, CCVariable<double>::getTypeDescription() );
  _source_grid_type = CC_SRC;

}

MoMICSoot::~MoMICSoot()
{
      VarLabel::destroy(m_PAH0_src_label      );
      VarLabel::destroy(m_PAH1_src_label      );
      VarLabel::destroy(m_PAH2_src_label      );
      VarLabel::destroy(m_PAH3_src_label      );
      VarLabel::destroy(m_PAH4_src_label      );
      VarLabel::destroy(m_Msoot0_src_label    );
      VarLabel::destroy(m_Msoot1_src_label    );
      VarLabel::destroy(m_Msoot2_src_label    );
      VarLabel::destroy(m_Msoot3_src_label    );
      VarLabel::destroy(m_Msoot4_src_label    );
      VarLabel::destroy(m_Msoot5_src_label    );
      VarLabel::destroy(m_Msurface_src_label  );
      VarLabel::destroy(m_balance_src_label  );
}
//---------------------------------------------------------------------------
// Method: Problem Setup
//---------------------------------------------------------------------------
void
MoMICSoot::problemSetup(const ProblemSpecP& inputdb)
{

  ProblemSpecP db = inputdb;

  db->getWithDefault("mix_mol_weight_label", m_mix_mol_weight_name,   "mixture_molecular_weight");
  db->getWithDefault("Msoot0_label",         m_Msoot0_name,           "Msoot0");
  db->getWithDefault("Msoot1_label",         m_Msoot1_name,           "Msoot1");
  db->getWithDefault("Msoot2_label",         m_Msoot2_name,           "Msoot2");
  db->getWithDefault("Msoot3_label",         m_Msoot3_name,           "Msoot3");
  db->getWithDefault("Msoot4_label",         m_Msoot4_name,           "Msoot4");
  db->getWithDefault("Msoot5_label",         m_Msoot5_name,           "Msoot5");
  db->getWithDefault("Msurface_label",       m_Msurface_name,         "Msurface");
  db->getWithDefault("PAH0_label",           m_PAH0_name,             "PAH0");
  db->getWithDefault("PAH1_label",           m_PAH1_name,             "PAH1");
  db->getWithDefault("PAH2_label",           m_PAH2_name,             "PAH2");
  db->getWithDefault("PAH3_label",           m_PAH3_name,             "PAH3");
  db->getWithDefault("PAH4_label",           m_PAH4_name,             "PAH4");
  db->getWithDefault("o2_label",             m_O2_name,               "O2");
  db->getWithDefault("oh_label",             m_OH_name,               "OH");
  db->getWithDefault("co2_label",	           m_CO2_name,		          "CO2");
  db->getWithDefault("h2o_label",            m_H2O_name,              "H2O");
  db->getWithDefault("h_label",              m_H_name,                "H");
  db->getWithDefault("h2_label",             m_H2_name,               "H2");
  db->getWithDefault("c2h2_label",           m_C2H2_name,             "C2H2");
  db->getWithDefault("pyrene_label",         m_pyrene_name,           "PYRENE");
  db->getWithDefault("density_label",        m_rho_name,              "density");
  db->getWithDefault("temperature_label",    m_temperature_name,      "radiation_temperature");

  db->findBlock("PAH0_src")         ->getAttribute( "label", m_PAH0_src_name );
  db->findBlock("PAH1_src")         ->getAttribute( "label", m_PAH1_src_name );
  db->findBlock("PAH2_src")         ->getAttribute( "label", m_PAH2_src_name );
  db->findBlock("PAH3_src")         ->getAttribute( "label", m_PAH3_src_name );
  db->findBlock("PAH4_src")         ->getAttribute( "label", m_PAH4_src_name );
  db->findBlock("Msoot0_src")       ->getAttribute( "label", m_Msoot0_src_name );
  db->findBlock("Msoot1_src")       ->getAttribute( "label", m_Msoot1_src_name );
  db->findBlock("Msoot2_src")       ->getAttribute( "label", m_Msoot2_src_name );
  db->findBlock("Msoot3_src")       ->getAttribute( "label", m_Msoot3_src_name );
  db->findBlock("Msoot4_src")       ->getAttribute( "label", m_Msoot4_src_name );
  db->findBlock("Msoot5_src")       ->getAttribute( "label", m_Msoot5_src_name );
  db->findBlock("Msurface_src")     ->getAttribute( "label", m_Msurface_src_name );
  db->findBlock("mass_balance_src") ->getAttribute( "label", m_balance_name );

  // Since we are producing multiple sources, we load each name into this std::vector
  // so that we can do error checking upon src term retrieval.
  _mult_srcs.push_back( m_PAH0_src_name );
  _mult_srcs.push_back( m_PAH1_src_name );
  _mult_srcs.push_back( m_PAH2_src_name );
  _mult_srcs.push_back( m_PAH3_src_name );
  _mult_srcs.push_back( m_PAH4_src_name );
  _mult_srcs.push_back( m_Msoot0_src_name );
  _mult_srcs.push_back( m_Msoot1_src_name );
  _mult_srcs.push_back( m_Msoot2_src_name );
  _mult_srcs.push_back( m_Msoot3_src_name );
  _mult_srcs.push_back( m_Msoot4_src_name );
  _mult_srcs.push_back( m_Msoot5_src_name );
  _mult_srcs.push_back( m_Msurface_src_name );
  _mult_srcs.push_back( m_balance_name );

  m_PAH0_src_label       = VarLabel::create( m_PAH0_src_name,     CCVariable<Vector>::getTypeDescription() );
  m_PAH1_src_label       = VarLabel::create( m_PAH1_src_name,     CCVariable<Vector>::getTypeDescription() );
  m_PAH2_src_label       = VarLabel::create( m_PAH2_src_name,     CCVariable<Vector>::getTypeDescription() );
  m_PAH3_src_label       = VarLabel::create( m_PAH3_src_name,     CCVariable<Vector>::getTypeDescription() );
  m_PAH4_src_label       = VarLabel::create( m_PAH4_src_name,     CCVariable<Vector>::getTypeDescription() );
  m_Msoot0_src_label     = VarLabel::create( m_Msoot0_src_name,   CCVariable<Vector>::getTypeDescription() );
  m_Msoot1_src_label     = VarLabel::create( m_Msoot1_src_name,   CCVariable<Vector>::getTypeDescription() );
  m_Msoot2_src_label     = VarLabel::create( m_Msoot2_src_name,   CCVariable<Vector>::getTypeDescription() );
  m_Msoot3_src_label     = VarLabel::create( m_Msoot3_src_name,   CCVariable<Vector>::getTypeDescription() );
  m_Msoot4_src_label     = VarLabel::create( m_Msoot4_src_name,   CCVariable<Vector>::getTypeDescription() );
  m_Msoot5_src_label     = VarLabel::create( m_Msoot5_src_name,   CCVariable<Vector>::getTypeDescription() );
  m_Msurface_src_label   = VarLabel::create( m_Msurface_src_name, CCVariable<Vector>::getTypeDescription() );
  m_balance_src_label    = VarLabel::create( m_balance_name,      CCVariable<Vector>::getTypeDescription() );

  ChemHelper& helper = ChemHelper::self();
  helper.add_lookup_species( m_O2_name );
  helper.add_lookup_species( m_OH_name );
  helper.add_lookup_species( m_rho_name );
  helper.add_lookup_species( m_CO2_name );
  helper.add_lookup_species( m_H2O_name );
  helper.add_lookup_species( m_H_name   );
  helper.add_lookup_species( m_H2_name  );
  helper.add_lookup_species( m_C2H2_name);
  helper.add_lookup_species( m_pyrene_name);
  helper.add_lookup_species( m_mix_mol_weight_name );
  //_field_labels->add_species( m_temperature_name );

}
//---------------------------------------------------------------------------
// Method: Schedule the calculation of the source term
//---------------------------------------------------------------------------
void
MoMICSoot::sched_computeSource( const LevelP& level, SchedulerP& sched, int timeSubStep )
{

  std::string taskname = "MoMICSoot::eval";
  Task* tsk = scinew Task(taskname, this, &MoMICSoot::computeSource, timeSubStep);
  Task::WhichDW which_dw;
  if (timeSubStep == 0) {
    tsk->computes(m_PAH0_src_label);
    tsk->computes(m_PAH1_src_label);
    tsk->computes(m_PAH2_src_label);
    tsk->computes(m_PAH3_src_label);
    tsk->computes(m_PAH4_src_label);
    tsk->computes(m_Msoot0_src_label);
    tsk->computes(m_Msoot1_src_label);
    tsk->computes(m_Msoot2_src_label);
    tsk->computes(m_Msoot3_src_label);
    tsk->computes(m_Msoot4_src_label);
    tsk->computes(m_Msoot5_src_label);
    tsk->computes(m_Msurface_src_label);
    tsk->computes(m_balance_src_label);
    which_dw = Task::OldDW;
  } else {
    which_dw = Task::NewDW;
    tsk->modifies(m_PAH0_src_label);
    tsk->modifies(m_PAH1_src_label);
    tsk->modifies(m_PAH2_src_label);
    tsk->modifies(m_PAH3_src_label);
    tsk->modifies(m_PAH4_src_label);
    tsk->modifies(m_Msoot0_src_label);
    tsk->modifies(m_Msoot1_src_label);
    tsk->modifies(m_Msoot2_src_label);
    tsk->modifies(m_Msoot3_src_label);
    tsk->modifies(m_Msoot4_src_label);
    tsk->modifies(m_Msoot5_src_label);
    tsk->modifies(m_Msurface_src_label);
    tsk->modifies(m_balance_src_label);
  }
  // resolve some labels:
  m_mix_mol_weight_label  = VarLabel::find( m_mix_mol_weight_name);
  m_PAH0_label            = VarLabel::find( m_PAH0_name);
  m_PAH1_label            = VarLabel::find( m_PAH1_name);
  m_PAH2_label            = VarLabel::find( m_PAH2_name);
  m_PAH3_label            = VarLabel::find( m_PAH3_name);
  m_PAH4_label            = VarLabel::find( m_PAH4_name);
  m_Msoot0_label          = VarLabel::find( m_Msoot0_name);
  m_Msoot1_label          = VarLabel::find( m_Msoot1_name);
  m_Msoot2_label          = VarLabel::find( m_Msoot2_name);
  m_Msoot3_label          = VarLabel::find( m_Msoot3_name);
  m_Msoot4_label          = VarLabel::find( m_Msoot4_name);
  m_Msoot5_label          = VarLabel::find( m_Msoot5_name);
  m_Msurface_label        = VarLabel::find( m_Msurface_name);
  m_o2_label              = VarLabel::find( m_O2_name);
  m_oh_label              = VarLabel::find( m_OH_name);
  m_co2_label             = VarLabel::find( m_CO2_name);
  m_h2o_label             = VarLabel::find( m_H2O_name);
  m_h_label               = VarLabel::find( m_H_name);
  m_h2_label              = VarLabel::find( m_H2_name);
  m_c2h2_label            = VarLabel::find( m_C2H2_name);
  m_pyrene_label          = VarLabel::find( m_pyrene_name);
  m_temperature_label     = VarLabel::find( m_temperature_name);
  m_rho_label             = VarLabel::find( m_rho_name);

  tsk->requires( which_dw, m_mix_mol_weight_label,               Ghost::None, 0 );
  tsk->requires( which_dw, m_PAH0_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_PAH1_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_PAH2_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_PAH3_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_PAH4_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_Msoot0_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_Msoot1_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_Msoot2_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_Msoot3_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_Msoot4_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_Msoot5_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_Msurface_label,                     Ghost::None, 0 );
  tsk->requires( which_dw, m_o2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_oh_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_co2_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_h2o_label,                          Ghost::None, 0 );
  tsk->requires( which_dw, m_h_label,                            Ghost::None, 0 );
  tsk->requires( which_dw, m_h2_label,                           Ghost::None, 0 );
  tsk->requires( which_dw, m_c2h2_label,                         Ghost::None, 0 );
  tsk->requires( which_dw, m_pyrene_label,                       Ghost::None, 0 );
  tsk->requires( which_dw, m_temperature_label,                  Ghost::None, 0 );
  tsk->requires( which_dw, m_rho_label,                          Ghost::None, 0 );

  sched->addTask(tsk, level->eachPatch(), _materialManager->allMaterials( "Arches" ));
  
  //get the system pressure:
  ChemHelper& helper = ChemHelper::self();
  ChemHelper::TableConstantsMapType tab_constants = helper.get_table_constants();
  auto i_press = tab_constants->find("Pressure");
  if ( i_press != tab_constants->end() ){
    m_sys_pressure = i_press->second;
  } else {
    m_sys_pressure = 101325.0; //otherise assume atmospheric
  }

}
//---------------------------------------------------------------------------
// Method: Actually compute the source term
//---------------------------------------------------------------------------
void
MoMICSoot::computeSource( const ProcessorGroup* pc,
                          const PatchSubset*    patches,
                          const MaterialSubset* matls,
                          DataWarehouse*  old_dw,
                          DataWarehouse*  new_dw,
                          int timeSubStep )
{
  //patch loop
  for (int p=0; p <patches->size(); p++){

    Ghost::GhostType    gn  =Ghost::None;
    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> PAH0_src;
    CCVariable<double> PAH1_src;
    CCVariable<double> PAH2_src;
    CCVariable<double> PAH3_src;
    CCVariable<double> PAH4_src;
    CCVariable<double> Msoot0_src;
    CCVariable<double> Msoot1_src;
    CCVariable<double> Msoot2_src;
    CCVariable<double> Msoot3_src;
    CCVariable<double> Msoot4_src;
    CCVariable<double> Msoot5_src;
    CCVariable<double> Msurface_src;
    CCVariable<double> balance_src;

    constCCVariable<double> mix_mol_weight;
    constCCVariable<double> PAH0;
    constCCVariable<double> PAH1;
    constCCVariable<double> PAH2;
    constCCVariable<double> PAH3;
    constCCVariable<double> PAH4;
    constCCVariable<double> Msoot0;
    constCCVariable<double> Msoot1;
    constCCVariable<double> Msoot2;
    constCCVariable<double> Msoot3;
    constCCVariable<double> Msoot4;
    constCCVariable<double> Msoot5;
    constCCVariable<double> Msurface;
    constCCVariable<double> O2;
    constCCVariable<double> OH;
    constCCVariable<double> CO2;
    constCCVariable<double> H2O;
    constCCVariable<double> H;
    constCCVariable<double> H2;
    constCCVariable<double> C2H2;
    constCCVariable<double> pyrene;
    constCCVariable<double> rho;
    constCCVariable<double> temperature; 
    
    DataWarehouse* which_dw;
    if ( timeSubStep == 0 ){
        which_dw = old_dw;
        new_dw->allocateAndPut( PAH0_src,        m_PAH0_src_label,        matlIndex, patch );
        new_dw->allocateAndPut( PAH1_src,        m_PAH1_src_label,        matlIndex, patch );
        new_dw->allocateAndPut( PAH2_src,        m_PAH2_src_label,        matlIndex, patch );
        new_dw->allocateAndPut( PAH3_src,        m_PAH3_src_label,        matlIndex, patch );
        new_dw->allocateAndPut( PAH4_src,        m_PAH4_src_label,        matlIndex, patch );
        new_dw->allocateAndPut( Msoot0_src,      m_Msoot0_src_label,      matlIndex, patch );
        new_dw->allocateAndPut( Msoot1_src,      m_Msoot1_src_label,      matlIndex, patch );
        new_dw->allocateAndPut( Msoot2_src,      m_Msoot2_src_label,      matlIndex, patch );
        new_dw->allocateAndPut( Msoot3_src,      m_Msoot3_src_label,      matlIndex, patch );
        new_dw->allocateAndPut( Msoot4_src,      m_Msoot4_src_label,      matlIndex, patch );
        new_dw->allocateAndPut( Msoot5_src,      m_Msoot5_src_label,      matlIndex, patch );
        new_dw->allocateAndPut( Msurface_src,    m_Msurface_src_label,    matlIndex, patch );
        new_dw->allocateAndPut( balance_src,     m_balance_src_label,     matlIndex, patch );
        PAH0_src.initialize(0.0);
        PAH1_src.initialize(0.0);
        PAH2_src.initialize(0.0);
        PAH3_src.initialize(0.0);
        PAH4_src.initialize(0.0);
        Msoot0_src.initialize(0.0);
        Msoot1_src.initialize(0.0);
        Msoot2_src.initialize(0.0);
        Msoot3_src.initialize(0.0);
        Msoot4_src.initialize(0.0);
        Msoot5_src.initialize(0.0);
        Msurface_src.initialize(0.0);
        balance_src.initialize(0.0);
    } else {
        which_dw = new_dw;
        new_dw->getModifiable( PAH0_src,     m_PAH0_src_label,        matlIndex, patch );
        new_dw->getModifiable( PAH1_src,     m_PAH1_src_label,        matlIndex, patch );
        new_dw->getModifiable( PAH2_src,     m_PAH2_src_label,        matlIndex, patch );
        new_dw->getModifiable( PAH3_src,     m_PAH3_src_label,        matlIndex, patch );
        new_dw->getModifiable( PAH4_src,     m_PAH4_src_label,        matlIndex, patch );
        new_dw->getModifiable( Msoot0_src,   m_Msoot0_src_label,      matlIndex, patch );
        new_dw->getModifiable( Msoot1_src,   m_Msoot1_src_label,      matlIndex, patch );
        new_dw->getModifiable( Msoot2_src,   m_Msoot2_src_label,      matlIndex, patch );
        new_dw->getModifiable( Msoot3_src,   m_Msoot3_src_label,      matlIndex, patch );
        new_dw->getModifiable( Msoot4_src,   m_Msoot4_src_label,      matlIndex, patch );
        new_dw->getModifiable( Msoot5_src,   m_Msoot5_src_label,      matlIndex, patch );
        new_dw->getModifiable( Msurface_src, m_Msurface_src_label,    matlIndex, patch );
        new_dw->getModifiable( balance_src,  m_balance_src_label,     matlIndex, patch );
    }

    which_dw->get( mix_mol_weight , m_mix_mol_weight_label , matlIndex , patch , gn, 0 );
    which_dw->get( PAH0           , m_PAH0_label           , matlIndex , patch , gn, 0 );
    which_dw->get( PAH1           , m_PAH1_label           , matlIndex , patch , gn, 0 );
    which_dw->get( PAH2           , m_PAH2_label           , matlIndex , patch , gn, 0 );
    which_dw->get( PAH3           , m_PAH3_label           , matlIndex , patch , gn, 0 );
    which_dw->get( PAH4           , m_PAH4_label           , matlIndex , patch , gn, 0 );
    which_dw->get( Msoot0         , m_Msoot0_label         , matlIndex , patch , gn, 0 );
    which_dw->get( Msoot1         , m_Msoot1_label         , matlIndex , patch , gn, 0 );
    which_dw->get( Msoot2         , m_Msoot2_label         , matlIndex , patch , gn, 0 );
    which_dw->get( Msoot3         , m_Msoot3_label         , matlIndex , patch , gn, 0 );
    which_dw->get( Msoot4         , m_Msoot4_label         , matlIndex , patch , gn, 0 );
    which_dw->get( Msoot5         , m_Msoot5_label         , matlIndex , patch , gn, 0 );
    which_dw->get( Msurface       , m_Msurface_label       , matlIndex , patch , gn, 0 );
    which_dw->get( O2             , m_o2_label             , matlIndex , patch , gn, 0 );
    which_dw->get( OH             , m_oh_label             , matlIndex , patch , gn, 0 );
    which_dw->get( CO2            , m_co2_label            , matlIndex , patch , gn, 0 );
    which_dw->get( H2O            , m_h2o_label            , matlIndex , patch , gn, 0 );
    which_dw->get( H              , m_h_label              , matlIndex , patch , gn, 0 );
    which_dw->get( H2             , m_h2_label             , matlIndex , patch , gn, 0 );
    which_dw->get( C2H2           , m_c2h2_label           , matlIndex , patch , gn, 0 );
    which_dw->get( pyrene         , m_pyrene_label         , matlIndex , patch , gn, 0 );
    which_dw->get( temperature    , m_temperature_label    , matlIndex , patch , gn, 0 );
    which_dw->get( rho            , m_rho_label            , matlIndex , patch , gn, 0 );

    /// Obtain time-step length
    delt_vartype DT;
    old_dw->get( DT,_field_labels->d_delTLabel);
    const double delta_t = DT;

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++) {
        IntVector c = *iter;
        std::vector<double> PAHbin(5);
        PAHbin[0] = 200.0/Na;
        PAHbin[1] = 350.0/Na;
        PAHbin[2] = 500.0/Na;
        PAHbin[3] = 850.0/Na;
        PAHbin[4] = 1200.0/Na;
        std::vector<double> PAH(5);
        PAH[0] = (PAH0[c] > 0.0) ? PAH0[c]*rho[c]/PAHbin[0] : 0.0;
        PAH[1] = (PAH1[c] > 0.0) ? PAH1[c]*rho[c]/PAHbin[1] : 0.0;
        PAH[2] = (PAH2[c] > 0.0) ? PAH2[c]*rho[c]/PAHbin[2] : 0.0;
        PAH[3] = (PAH3[c] > 0.0) ? PAH3[c]*rho[c]/PAHbin[3] : 0.0;
        PAH[4] = (PAH4[c] > 0.0) ? PAH4[c]*rho[c]/PAHbin[4] : 0.0;
        std::vector<double> Msoot(6);
        Msoot[0] = (Msoot0[c] > 0.0) ? abs(Msoot0[c]*rho[c]) : 0.0;
        Msoot[1] = (Msoot1[c] > 0.0) ? abs(Msoot1[c]*rho[c]) : 0.0;
        Msoot[2] = (Msoot2[c] > 0.0) ? abs(Msoot2[c]*rho[c]) : 0.0;
        Msoot[3] = (Msoot3[c] > 0.0) ? abs(Msoot3[c]*rho[c]) : 0.0;
        Msoot[4] = (Msoot4[c] > 0.0) ? abs(Msoot4[c]*rho[c]) : 0.0;
        Msoot[5] = (Msoot5[c] > 0.0) ? abs(Msoot5[c]*rho[c]) : 0.0;
        if (std::any_of(Msoot.begin(),Msoot.end(),[](double i){return i<=0;})) {
          for (int i=0;i<6;i++) Msoot[i] = 0;
        }
        const double T = temperature[c];
        const double P = m_sys_pressure;
        const double dt = delta_t;

        const double PC2H2  = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (C2H2[c]*1.0/(mix_mol_weight[c]*26.0))*P : 0.0;
        const double PH     = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (H[c]   *1.0/(mix_mol_weight[c]*1.0))*P  : 0.0;
        const double PH2    = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (H2[c]  *1.0/(mix_mol_weight[c]*2.0))*P  : 0.0;
        const double cH2 = abs(PH2/(Rgas*T));
        const double PO2    = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (O2[c]  *1.0/(mix_mol_weight[c]*32.0))*P : 0.0;
        const double POH    = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (OH[c]  *1.0/(mix_mol_weight[c]*17.0))*P : 0.0;
        const double PH2O   = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (H2O[c] *1.0/(mix_mol_weight[c]*18.0))*P : 0.0;
        const double PCO2   = ( mix_mol_weight[c] > 1.0e-10 ) ?
          (CO2[c] *1.0/(mix_mol_weight[c]*44.0))*P : 0.0;
        //const double Pyrene = (pyrene[c] > 1.0e-10) ? pyrene[c]*rho[c] : 0.0; // UNUSED VAR?
        //PAH[0] = PAH[0] + Pyrene*Na/202.25; <!--Disabled as equilibrium tables always break this term-->
        
        double d  = (Msurface[c]>0.0) ? log10(Msurface[c]*rho[c]/Msoot[0])/log10(Msoot[1]/Msoot[0]) : 2.0/3.0;
        d = ( d < 2.0/3.0 ) ? 2.0/3.0 : ( d > 1.0 ) ? 1.0 : d;
        const double Ca = (3.0-3.0*d)+(3.0*d-2.0)*Cd;
        //-------- Precursors
        std::vector<double> PAHCoagulation(5);
        for (int i=0;i<5;i++) PAHCoagulation[i] = 0;
        std::vector<double> PAHDeposit(5);
        for (int i=0;i<5;i++) PAHDeposit[i] = 0;
        std::vector<double> PAHCracking(5);
        std::vector<double> PAHSurfaceGrowth(5);
        std::vector<double> PAH_src(5);
        double PhenolFrac,TolueneFrac,NapthyleneFrac,BenzeneFrac;
        std::vector<double> PAHMoments(2);
        
        for (int i=0;i<5;i++){
          // Nucleation
          if (PAH[i]>0){
            for (int j=0;j<5;j++) {
              if(PAH[j]>0.0 and PAH[i]>0.0) {
                double mu = (PAHbin[i]*PAHbin[j])/(PAHbin[i]+PAHbin[j]);
                double Di = Ch*pow(PAHbin[i],0.5);
                double Dj = Ch*pow(PAHbin[j],0.5);
                double beta = 2.2*pow(M_PI*kB*abs(T)/(2.0*mu),0.5)*pow(Di+Dj,2.0);
                PAHCoagulation[i] = PAHCoagulation[i]+beta*PAH[i]*PAH[j];
              }
            }
            // Cracking
            PhenolFrac = abs(tanh(5.73-0.00384*abs(T)-0.159*log10(PAH[i]))/6.0-0.218+0.0277*log10(PAH[i]));
            NapthyleneFrac= abs(tanh(-1.98+6.18E-4*abs(T)+0.124*log10(PAH[i])-0.00285*pow(log10(PAH[i]),2.0)+4.14E-7*pow(abs(T),2.0)-4.97E-5*abs(T)*log10(PAH[i]))/2.0-0.576+0.000233*abs(T)-1.69E-7*pow(abs(T),2.0));
            TolueneFrac = abs(tanh(17.3-0.00869*abs(T)-1.08*log10(PAH[i])+0.0199*pow(log10(PAH[i]),2.0)+0.000365*abs(T)*log10(PAH[i]))/3.0+0.000265*abs(T)-0.000111*pow(log10(PAH[i]),2.0)-9.32E-6*abs(T)*log10(PAH[i]));
            BenzeneFrac = 1-PhenolFrac-NapthyleneFrac-TolueneFrac;
            PAHCracking[i] = (50.0/128.0*1.58e12*exp(-3.24e8/Rgas/abs(T))*NapthyleneFrac*pow(cH2,0.4)+
                              14.0/92.0*1.04e12*exp(-2.47e8/Rgas/abs(T))*TolueneFrac*pow(cH2,0.5)+
                              31.1/94.0*1.0e7*exp(-1.0e8/Rgas/abs(T))*PhenolFrac+
                              93.4/94.0*1.0e8*exp(-1.0e8/Rgas/abs(T))*PhenolFrac+
                              4.4e8*exp(-2.2e8/Rgas/abs(T))*BenzeneFrac)*PAH[i];
            // Deposition
            PAHDeposit[i] = (Msoot[0] > 0.0) ? 2.2*pow(M_PI*kB*T/2.0,0.5)*PAH[i]*(pow(Ch,2.0)*pow(PAHbin[i],0.5)*Msoot[0]+2.0*Ch*Ca*Cs*MoMIC(1.0/3.0,Msoot)+pow(Ca*Cs,2.0)*pow(PAHbin[i],-0.5)*MoMIC(2.0/3.0,Msoot)) : 0;
            // Surface Reactions
            PAHMoments[0]= PAH[i];
            PAHMoments[1]= PAH[i]*PAHbin[i];
            PAHSurfaceGrowth[i] = 5e-20*(HACA(T,PC2H2,PH,PH2,PO2,POH,PH2O,PAHMoments)-Oxidation(T,PO2,POH)-Gasification(T,PCO2,PH2O))*PAH[i]/mC;
    
            PAH_src[i] = (PAHSurfaceGrowth[i]-PAHCoagulation[i]-PAHCracking[i]-PAHDeposit[i])*PAHbin[i];
          } else PAH_src[i] = 0.0;
        }
        if (PAH_src[0] < -(PAH[0]*PAHbin[0])/dt){
          PAH0_src[c] = -(PAH[0]*PAHbin[0])/dt;
          PAHCoagulation[0]   = PAHCoagulation[0]   * abs(PAH0_src[c]/PAH_src[0]);
          PAHDeposit[0]       = PAHDeposit[0]       * abs(PAH0_src[c]/PAH_src[0]);
          PAHCracking[0]      = PAHCracking[0]      * abs(PAH0_src[c]/PAH_src[0]);
          PAHSurfaceGrowth[0] = PAHSurfaceGrowth[0] * abs(PAH0_src[c]/PAH_src[0]);
        } else { PAH0_src[c] = PAH_src[0]; }
        if (PAH_src[1] < -(PAH[1]*PAHbin[1])/dt){
          PAH1_src[c] = -(PAH[1]*PAHbin[1])/dt;
          PAHCoagulation[1]   = PAHCoagulation[1]   * abs(PAH1_src[c]/PAH_src[1]);
          PAHDeposit[1]       = PAHDeposit[1]       * abs(PAH1_src[c]/PAH_src[1]);
          PAHCracking[1]      = PAHCracking[1]      * abs(PAH1_src[c]/PAH_src[1]);
          PAHSurfaceGrowth[1] = PAHSurfaceGrowth[1] * abs(PAH1_src[c]/PAH_src[1]);
        } else { PAH0_src[c] = PAH_src[0]; }
        if (PAH_src[2] < -(PAH[2]*PAHbin[2])/dt){
          PAH2_src[c] = -(PAH[2]*PAHbin[2])/dt;
          PAHCoagulation[2]   = PAHCoagulation[2]   * abs(PAH2_src[c]/PAH_src[2]);
          PAHDeposit[2]       = PAHDeposit[2]       * abs(PAH2_src[c]/PAH_src[2]);
          PAHCracking[2]      = PAHCracking[2]      * abs(PAH2_src[c]/PAH_src[2]);
          PAHSurfaceGrowth[2] = PAHSurfaceGrowth[2] * abs(PAH2_src[c]/PAH_src[2]);
        } else { PAH0_src[c] = PAH_src[0]; }
        if (PAH_src[3] < -(PAH[3]*PAHbin[3])/dt){
          PAH3_src[c] = -(PAH[3]*PAHbin[3])/dt;
          PAHCoagulation[3]   = PAHCoagulation[3]   * abs(PAH3_src[c]/PAH_src[3]);
          PAHDeposit[3]       = PAHDeposit[3]       * abs(PAH3_src[c]/PAH_src[3]);
          PAHCracking[3]      = PAHCracking[3]      * abs(PAH3_src[c]/PAH_src[3]);
          PAHSurfaceGrowth[3] = PAHSurfaceGrowth[3] * abs(PAH3_src[c]/PAH_src[3]);
        } else { PAH0_src[c] = PAH_src[0]; }
        if (PAH_src[4] < -(PAH[4]*PAHbin[4])/dt){
          PAH4_src[c] = -(PAH[4]*PAHbin[4])/dt;
          PAHCoagulation[4]   = PAHCoagulation[4]   * abs(PAH4_src[c]/PAH_src[4]);
          PAHDeposit[4]       = PAHDeposit[4]       * abs(PAH4_src[c]/PAH_src[4]);
          PAHCracking[4]      = PAHCracking[4]      * abs(PAH4_src[c]/PAH_src[4]);
          PAHSurfaceGrowth[4] = PAHSurfaceGrowth[4] * abs(PAH4_src[c]/PAH_src[4]);
        } else { PAH0_src[c] = PAH_src[0]; }
        
        //---------- Soot

        //------ Nucleation 
        std::vector<double> SootNucleationRate(6);
        for (int i=0;i<6;i++){
          SootNucleationRate[i] = 0;
          for (int j=0;j<5;j++){
            SootNucleationRate[i] = SootNucleationRate[i] + 0.5*PAHCoagulation[j]*pow(2.0*PAHbin[j],i);
          }
        }
        //----- Deposition
        std::vector<double> SootPAHGrowthRate(6);
        SootPAHGrowthRate[0] = 0;
        for (int i=1;i<6;i++){
          SootPAHGrowthRate[i] = 0;
          for (int j=0;j<5;j++){
            SootPAHGrowthRate[i] = SootPAHGrowthRate[i] + PAHDeposit[j]*pow(PAHbin[j],i);
          }
        }
        std::vector<double> SootCoagulationRate(6);
        std::vector<double> SootSurfaceGrowthRate(6);
        double SurfaceNucleate = pow(m0,2.0/3.0)*SootNucleationRate[0];
        std::vector<double> PAHDeposition(6);
        std::vector<int> x(6);
        for (int i=1;i<6;i++) x[i-1] = i;
        for (int i=1;i<6;i++) PAHDeposition[i-1] = log10(SootPAHGrowthRate[i-1]);
        double SurfaceDeposition = pow(10,LagrangianInterpolation(d,x,SootPAHGrowthRate));
        double SurfaceSurfaceGrowth;
        double SurfaceCoagulation;
        
        std::vector<double> Msoot_new(6);
        double Msurface_old,Msurface_new;
        Msurface_old = Msurface[c]*rho[c];

        //----- Loop through the whole moments
        for (int i=0;i<6;i++){
          if (Msoot[0]<=0){
            SootCoagulationRate[i]   = 0;
            SootPAHGrowthRate[i]     = 0;
            SootSurfaceGrowthRate[i] = 0;
          } else{
            //----- Coagulation
            SootCoagulationRate[i] = SootCoagulation(T,P,Msoot,i,d,dt);
            //----- Surface Reactions
            SootSurfaceGrowthRate[i] = SurfaceGrowth(T,PC2H2,PH,PH2,PO2,POH,PH2O,PCO2,Msoot,i,d);
          }
          Msoot_new[i] = Msoot[i] + (SootNucleationRate[i]+SootCoagulationRate[i]+SootSurfaceGrowthRate[i]+SootPAHGrowthRate[i])*dt;
        }
        //----- Surface Moment
        if (Msoot[0]<=0) {
          SurfaceCoagulation    = 0;
          SurfaceDeposition     = 0;
          SurfaceSurfaceGrowth  = 0;
        } else{
          SurfaceCoagulation = SootSurfaceCoagulation(T,P,Msoot,d);
          SurfaceSurfaceGrowth = M_PI*pow(Cs,2.0)*pow(m0,2.0/3.0-d)*(HACA(T,PC2H2,PH,PH2,PO2,POH,PH2O,Msoot)/(2.0*mC)*(f_grid_surface_reactions(d,2.0*mC,Msoot)-MoMIC(2.0*d,Msoot))-(f_grid_surface_reactions(d,mC,Msoot)-MoMIC(2.0*d,Msoot))/mC*(Oxidation(T,PO2,POH)+Gasification(T,PCO2,PH2O)));
        }
        Msurface_new = Msurface_old + (SurfaceNucleate+SurfaceSurfaceGrowth+SurfaceDeposition+SurfaceCoagulation)*dt;

        if (std::all_of(Msoot_new.begin(),Msoot_new.end(),[](double i){return i>0;}) and Msoot_new[1] > 1E-12){
          Msoot0_src[c] = (Msoot_new[0]-Msoot[0])/dt;
          Msoot1_src[c] = (Msoot_new[1]-Msoot[1])/dt;
          Msoot2_src[c] = (Msoot_new[2]-Msoot[2])/dt;
          Msoot3_src[c] = (Msoot_new[3]-Msoot[3])/dt;
          Msoot4_src[c] = (Msoot_new[4]-Msoot[4])/dt;
          Msoot5_src[c] = (Msoot_new[5]-Msoot[5])/dt;
          Msurface_src[c] = (Msurface_new-Msurface[c]*rho[c])/dt;
        } else {
          Msoot0_src[c] = -Msoot0[c]*rho[c]/dt;
          Msoot1_src[c] = -Msoot1[c]*rho[c]/dt;
          Msoot2_src[c] = -Msoot2[c]*rho[c]/dt;
          Msoot3_src[c] = -Msoot3[c]*rho[c]/dt;
          Msoot4_src[c] = -Msoot4[c]*rho[c]/dt;
          Msoot5_src[c] = -Msoot5[c]*rho[c]/dt;
          Msurface_src[c] = -(Msurface[c]*rho[c])/dt;
        }

        // --------- Gas source
        balance_src[c] = -Msoot1_src[c]-PAH0_src[c]-PAH1_src[c]-PAH2_src[c]-PAH3_src[c]-PAH4_src[c];
    }
  } 
}
//---------------------------------------------------------------------------
// Method: Schedule initialization
//---------------------------------------------------------------------------
void
MoMICSoot::sched_initialize( const LevelP& level, SchedulerP& sched )
{
    string taskname = "MoMICSoot::initialize";

    Task* tsk = scinew Task(taskname, this, &MoMICSoot::initialize);

    tsk->computes(m_PAH0_src_label);
    tsk->computes(m_PAH1_src_label);
    tsk->computes(m_PAH2_src_label);
    tsk->computes(m_PAH3_src_label);
    tsk->computes(m_PAH4_src_label);
    tsk->computes(m_Msoot0_src_label);
    tsk->computes(m_Msoot1_src_label);
    tsk->computes(m_Msoot2_src_label);
    tsk->computes(m_Msoot3_src_label);
    tsk->computes(m_Msoot4_src_label);
    tsk->computes(m_Msoot5_src_label);
    tsk->computes(m_Msurface_src_label);
    tsk->computes(m_balance_src_label);

    sched->addTask(tsk, level->eachPatch(),_materialManager->allMaterials( "Arches" ));

}
void
MoMICSoot::initialize( const ProcessorGroup* pc,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw )
{
  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = _materialManager->getMaterial( "Arches", archIndex)->getDWIndex();

    CCVariable<double> PAH0_src;
    CCVariable<double> PAH1_src;
    CCVariable<double> PAH2_src;
    CCVariable<double> PAH3_src;
    CCVariable<double> PAH4_src;
    CCVariable<double> Msoot0_src;
    CCVariable<double> Msoot1_src;
    CCVariable<double> Msoot2_src;
    CCVariable<double> Msoot3_src;
    CCVariable<double> Msoot4_src;
    CCVariable<double> Msoot5_src;
    CCVariable<double> Msurface_src;
    CCVariable<double> balance_src;

    new_dw->allocateAndPut( PAH0_src,        m_PAH0_src_label,        matlIndex, patch );
    new_dw->allocateAndPut( PAH1_src,        m_PAH1_src_label,        matlIndex, patch );
    new_dw->allocateAndPut( PAH2_src,        m_PAH2_src_label,        matlIndex, patch );
    new_dw->allocateAndPut( PAH3_src,        m_PAH3_src_label,        matlIndex, patch );
    new_dw->allocateAndPut( PAH4_src,        m_PAH4_src_label,        matlIndex, patch );
    new_dw->allocateAndPut( Msoot0_src,      m_Msoot0_src_label,      matlIndex, patch );
    new_dw->allocateAndPut( Msoot1_src,      m_Msoot1_src_label,      matlIndex, patch );
    new_dw->allocateAndPut( Msoot2_src,      m_Msoot2_src_label,      matlIndex, patch );
    new_dw->allocateAndPut( Msoot3_src,      m_Msoot3_src_label,      matlIndex, patch );
    new_dw->allocateAndPut( Msoot4_src,      m_Msoot4_src_label,      matlIndex, patch );
    new_dw->allocateAndPut( Msoot5_src,      m_Msoot5_src_label,      matlIndex, patch );
    new_dw->allocateAndPut( Msurface_src,    m_Msurface_src_label,    matlIndex, patch );
    new_dw->allocateAndPut( balance_src,     m_balance_src_label,     matlIndex, patch );

    PAH0_src.initialize(0.0);
    PAH1_src.initialize(0.0);
    PAH2_src.initialize(0.0);
    PAH3_src.initialize(0.0);
    PAH4_src.initialize(0.0);
    Msoot0_src.initialize(0.0);
    Msoot1_src.initialize(0.0);
    Msoot2_src.initialize(0.0);
    Msoot3_src.initialize(0.0);
    Msoot4_src.initialize(0.0);
    Msoot5_src.initialize(0.0);
    Msurface_src.initialize(0.0);
    balance_src.initialize(0.0);

  }
}
