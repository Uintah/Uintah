/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "Particles.h"

#include <cmath>
#include <iostream>

#include <tabprops/LagrangeInterpolant.h>

using namespace std;

//==============================================================================

/**
 * \class Mie
 * \brief Orion Sky Lawlor, olawlor@acm.org, 9/23/2000
 * Computes so-called "Mie scattering" for a homogenous
 * sphere of arbitrary size illuminated by coherent
 * harmonic radiation.
 */
class Mie
{
public:
  Mie( const double xt, const std::complex<double>& mt );
  ~Mie();

  void calcS(double theta);

  double getQext() const{ return qext; }
  double getQsca() const{ return qsca; }
  double getQabs() const{ return qext-qsca; }
  std::complex<double> getS1() const{ return s1; }
  std::complex<double> getS2() const{ return s2; }

  int getNmax() const{return n;}
  std::complex<double> getM() const{ return m; }
  double getX() const{ return x; }

private:
  const std::complex<double> m;
  const double x;
  int n;
  std::complex<double> mx;
  int Nstar;

  std::complex<double> *an;
  std::complex<double> *bn;
  std::complex<double> *rn;
  std::complex<double> ccotmx;
  double *chi;
  double *psi;
  double *pi;
  double qext;
  double qsca;
  double qabs;
  std::complex<double> s1;
  std::complex<double> s2;

  std::complex<double> Cot(std::complex<double> mx);

  void calcPi(double mu);
  double Tau(double mu,int n);
  void calcChi(double x,int n);
  void calcPsi(double x,int n);
  std::complex<double> Zeta(double x,int n);

  void calcR();
  void calcAnBn();

  void calcSi(double theta);
  void calcQscaQext();
};


Mie::Mie( const double xt, const complex<double>& mt )
: m(mt), x(xt)
{
  mx=m*x;
  Nstar=60000;
  n=x+4.0*pow(x,1.0/3.0)+2.0+10; // when to cut of the summations (term taken from Hong Du)
  an=new complex<double>[n];
  bn=new complex<double>[n];
  rn=new complex<double>[Nstar+1];
  chi=new double[n+1];
  psi=new double[n+1];
  pi=new double[n+1];
  calcChi(x,n);
  calcPsi(x,n);
  ccotmx=conj(Cot(conj(mx)));
  calcR();
  calcAnBn();
  calcQscaQext();
}

Mie::~Mie(){
  delete [] an;
  delete [] bn;
  delete [] rn;
  delete [] chi;
  delete [] psi;
  delete [] pi;
}


void Mie::calcS(double theta){
  calcSi(theta);
}

void Mie::calcPi(double mu){
  double s=0.0,t=0.0;

  pi[0]=0.0;
  pi[1]=1.0;

  for(int i=1;i<n;i++){
    s=mu*pi[i];
    t=s-pi[i-1];
    pi[i+1]=s+t+t/(double)i;
  }
}

double Mie::Tau(double mu,int n){
  if(n==0){
    return 0.0;
  }else{
    return n*(mu*pi[n]-pi[n-1])-pi[n-1];
  }
}

void Mie::calcChi(double x,int n){
  chi[0]=cos(x);
  chi[1]=chi[0]/x+sin(x);

  for(int i=1;i<n;i++){
    chi[i+1]=(2.0*i+1.0)*chi[i]/x-chi[i-1];
  }
}

void Mie::calcPsi(double x,int n){
  psi[0]=sin(x);
  psi[1]=psi[0]/x-cos(x);

  for(int i=1;i<n;i++){
    psi[i+1]=(2.0*i+1.0)*psi[i]/x-psi[i-1];
  }
}

complex<double> Mie::Zeta(double x,int n){
  return complex<double>(psi[n],chi[n]);
}

complex<double> Mie::Cot(complex<double> mx){
  const double ctan=tan(real(mx));
  const double cexp=exp(-2.0*imag(mx));

  const complex<double> num   = complex<double>(0,1)+ctan-cexp*ctan+complex<double>(0,1.0)*cexp;
  const complex<double> denom = -(double)1.0+complex<double>(0,1.0)*ctan+complex<double>(0,1.0)*cexp*ctan+cexp;

  const double realnum=real(num);
  const double imagnum=imag(num);
  const double realdenom=real(denom);
  const double imagdenom=imag(denom);

  const double div=(realdenom*realdenom+imagdenom*imagdenom);

  const double realcot=((realnum*realdenom+imagnum*imagdenom)/div);
  const double imagcot=((imagnum*realdenom-realnum*imagdenom)/div);

  return complex<double>(realcot,imagcot);
}

void Mie::calcR(){ //downward recurrence
  rn[Nstar-1]=(double)(2.0*Nstar+1.0)/mx;

  for(int i=Nstar-1;i>=1;i--){
    rn[i-1]=(2.0*i+1.0)/mx-1.0/rn[i];
  }
}

void Mie::calcAnBn(){
  complex<double> rf;

  for(int nc=1;nc<=n;nc++){
    rf=(rn[nc-1]/m+(double)nc*(1.0-1.0/(m*m))/x);

    an[nc-1]=(rf*psi[nc]-psi[nc-1])/(rf*Zeta(x,nc)-Zeta(x,nc-1));
    bn[nc-1]=((complex<double>)rn[nc-1]*(complex<double>)m*psi[nc]-psi[nc-1])/((complex<double>)rn[nc-1]*(complex<double>)m*Zeta(x,nc)-Zeta(x,nc-1));
  }
}

void Mie::calcSi(double theta){
  double mu=cos(theta);
  double tau;
  double fn;

  s1=complex<double>(0,0);
  s2=complex<double>(0,0);

  calcPi(mu);

  for(int i=1;i<=n;i++){
    tau=Tau(mu,i);
    fn=(double)(2.0*i+1.0)/(double)(i*(i+1.0));
    s1=s1+fn*(an[i-1]*pi[i]+bn[i-1]*tau);
    s2=s2+fn*(an[i-1]*tau+bn[i-1]*pi[i]);
  }
}

void Mie::calcQscaQext(){
  qsca=0;
  qext=0;

  for(int i=1;i<=n;i++){
    qext=qext+(double)(2.0*i+1.0)*(an[i-1].real()+bn[i-1].real());
    qsca=qsca+(double)(2.0*i+1.0)*(pow(abs(an[i-1]),2)+pow(abs(bn[i-1]),2));
  }

  qext=qext*2.0/(x*x);
  qsca=qsca*2.0/(x*x);
}

//==============================================================================

ParticleRadCoeffs::ParticleRadCoeffs( const std::complex<double> refIndex )
{
  std::vector<double> absSpectralEff, scaSpectralEff, lambdas, raV;
  spectral_effs( absSpectralEff, scaSpectralEff, lambdas, raV, 1e-4, 1e-7, refIndex );

  std::vector<double> absSpectralCoeff;
  std::vector<double> scaSpectralCoeff;
  spectral_coeff( absSpectralCoeff, scaSpectralCoeff, absSpectralEff, scaSpectralEff, lambdas, raV );

  const int interpOrder = 1;
  const bool clipValues = true;

  interp2AbsSpectralCoeff_ = new LagrangeInterpolant2D( interpOrder, lambdas, raV, absSpectralCoeff, clipValues );
  interp2ScaSpectralCoeff_ = new LagrangeInterpolant2D( interpOrder, lambdas, raV, scaSpectralCoeff, clipValues );

  std::vector<double> absCoeffRoss, scaCoeffRoss, absCoeffPlanck, scaCoeffPlanck, tempV;

  mean_coeff( absCoeffPlanck, scaCoeffPlanck, absCoeffRoss, scaCoeffRoss, absSpectralEff, scaSpectralEff, lambdas, raV, tempV, 600, 2800 );

  interp2AbsPlanckCoeff_ = new LagrangeInterpolant2D( interpOrder, raV, tempV, absCoeffPlanck, clipValues );
  interp2ScaPlanckCoeff_ = new LagrangeInterpolant2D( interpOrder, raV, tempV, scaCoeffPlanck, clipValues );
  interp2AbsRossCoeff_   = new LagrangeInterpolant2D( interpOrder, raV, tempV, absCoeffRoss,   clipValues );
  interp2ScaRossCoeff_   = new LagrangeInterpolant2D( interpOrder, raV, tempV, scaCoeffRoss,   clipValues );
}

ParticleRadCoeffs::~ParticleRadCoeffs()
{
  delete interp2AbsSpectralCoeff_;
  delete interp2ScaSpectralCoeff_;
  delete interp2AbsPlanckCoeff_;
  delete interp2ScaPlanckCoeff_;
  delete interp2AbsRossCoeff_;
  delete interp2ScaRossCoeff_;
}

void
ParticleRadCoeffs::spectral_effs( std::vector<double>& AbsSpectralEff,
                                  std::vector<double>& ScaSpectralEff,
                                  std::vector<double>& lambdas,
                                  std::vector<double>& raV,
                                  const double ramax,
                                  const double ramin,
                                  const complex<double> refIndex )
{
  const double wl_max = 12.5e-6;  // (m)
  const double wl_min = 1e-7;     // (m)
  const int N=100;
  lambdas.resize(N);

  const double log_wl_max = log(wl_max);
  const double log_wl_min = log(wl_min);
  const double log_wl_step = (log_wl_max-log_wl_min)/N;

  int R= 10; //number of radii to be considered
  int NR = N*R;
  AbsSpectralEff.resize(NR);
  ScaSpectralEff.resize(NR);
  raV.resize(R);


  const double log_ramax = log(ramax);
  const double log_ramin = log(ramin);

  const double log_ra_step = (log_ramax-log_ramin)/R;

  int coeff = 0;
  for (int r=0;r<R;r++){
    raV[r]=log_ramin+r*log_ra_step;

    double sum00 =0;
    double wl_step;
    double x;
    for (int i=0;i<N;i++){
      lambdas[i]= log_wl_min+i*log_wl_step;
      x=2*M_PI*exp(raV[r])/exp(lambdas[i]);
      Mie mie2(x,refIndex );

      double qabs=mie2.getQabs();
      double qsca=mie2.getQsca();

      AbsSpectralEff[coeff] = qabs;
      ScaSpectralEff[coeff] = qsca;

      coeff+=1;
    }
  }
}

void
ParticleRadCoeffs::spectral_coeff( std::vector<double>& absSpectralCoeff,
                                   std::vector<double>& scaSpectralCoeff,
                                   const std::vector<double>& absSpectralEff,
                                   const std::vector<double>& scaSpectralEff,
                                   const std::vector<double>& lambdas,
                                   const std::vector<double>& raV )
{
  const int nL = lambdas.size();
  const int nR= raV.size(); //number of radii to be considered
  const int nRL = nR*nL;
  absSpectralCoeff.resize(nRL);
  scaSpectralCoeff.resize(nRL);

  for (int r=0;r<nR;r++){
    for (int i=0;i<nL;i++){
      // exp(r) because we have done a log-transform on the radii previously.
      absSpectralCoeff[r*nL+i] = M_PI*exp(raV[r])*exp(raV[r])*absSpectralEff[r*nL+i];
      scaSpectralCoeff[r*nL+i] = M_PI*exp(raV[r])*exp(raV[r])*scaSpectralEff[r*nL+i];
#     ifdef RADPROPS_DEBUG_INFO
      cout  << exp(lambdas[i]) << " " << exp(raV[r]) << " " << absSpectralCoeff[r*nL+i] << " " << scaSpectralCoeff[r*nL+i] << endl;
#     endif // RADPROPS_DEBUG_INFO
    }
  }
}



void
ParticleRadCoeffs::mean_coeff( std::vector<double>& absPlanckCoeff,
                               std::vector<double>& scaPlanckCoeff,
                               std::vector<double>& absRossCoeff,
                               std::vector<double>& scaRossCoeff,
                               const std::vector<double>& absSpectralEff,
                               const std::vector<double>& scaSpectralEff,
                               const std::vector<double>& lambdas,
                               const std::vector<double>& raV,
                               std::vector<double>& tempV,
                               const double minT,
                               const double maxT )
{
  const int nR = raV.size(); //number of radii to be considered

  const double stepT=600.0;
  const int nTemp= int((maxT-minT)/stepT)+1; //number of temperatures to be considered
  tempV.resize(nTemp);

  const int nTR=nTemp*nR;
  absPlanckCoeff.resize(nTR);
  scaPlanckCoeff.resize(nTR);
  absRossCoeff.resize(nTR);
  scaRossCoeff.resize(nTR);

  const int L = lambdas.size();

  int coeff =0;

  for (int tt=0;tt<nTemp; tt++){
    tempV[tt]=minT+(tt)*stepT;
    for (int r=0;r<nR;r++){

      double integralAbsRoss=0;
      double integralScaRoss=0;
      double integralAbsPlanck=0;
      double integralScaPlanck=0;
      for (int i=0;i<L-1;i++){

        const double c22 = 0.0144;
        const double xsi0=c22/(exp(lambdas[i])*tempV[tt]);
        const double xsi1=c22/(exp(lambdas[i+1])*tempV[tt]);
        const double xsi_step=xsi0-xsi1;

        const double expxsi1 = exp(xsi1);
        const double expxsi0 = exp(xsi0);
        const double xsi1_3 = xsi1*xsi1*xsi1;
        const double xsi0_3 = xsi0*xsi0*xsi0;
        const double xsi1_4 = xsi1_3 * xsi1;
        const double xsi0_4 = xsi0_3 * xsi0;

        integralAbsPlanck += xsi_step*( xsi1_3/(expxsi1-1)*absSpectralEff[r*L+i+1] + xsi0_3*1/(expxsi0-1)*absSpectralEff[r*L+i] );
        integralScaPlanck += xsi_step*( xsi1_3/(expxsi1-1)*scaSpectralEff[r*L+i+1] + xsi0_3*1/(expxsi0-1)*scaSpectralEff[r*L+i] );

        integralAbsRoss += xsi_step*( xsi1_4*expxsi1 / ( (expxsi1-1)*(expxsi1-1) ) / absSpectralEff[r*L+i+1] + xsi0_4*expxsi0/((expxsi0-1)*(expxsi0-1))/absSpectralEff[r*L+i] );
        integralScaRoss += xsi_step*( xsi1_4*expxsi1 / ( (expxsi1-1)*(expxsi1-1) ) / scaSpectralEff[r*L+i+1] + xsi0_4*expxsi0/((expxsi0-1)*(expxsi0-1))/scaSpectralEff[r*L+i] );
      }
      const double expR = exp(raV[r]);
      const double pi4 = M_PI*M_PI*M_PI*M_PI;
      absPlanckCoeff[coeff] = 15*M_PI*expR*expR/      pi4*integralAbsPlanck/2.0;
      scaPlanckCoeff[coeff] = 15*M_PI*expR*expR/      pi4*integralScaPlanck/2.0;
      absRossCoeff  [coeff] = pi4*M_PI*expR*expR/(15.0*integralAbsRoss/2.0);
      scaRossCoeff  [coeff] = pi4*M_PI*expR*expR/(15.0*integralScaRoss/2.0);

#     ifdef RADPROPS_DEBUG_INFO
      cout << tempV[tt] << " " << exp(raV[r]) << " " << absPlanckCoeff[coeff] << " " << scaPlanckCoeff[coeff]<< " " << absRossCoeff[coeff] << " " << scaRossCoeff[coeff] << endl;
#     endif // RADPROPS_DEBUG_INFO
      coeff+=1;
    }
  }
}

double
ParticleRadCoeffs::abs_spectral_coeff( const double wavelength, const double r) const{
  double indepVars [] = {log(wavelength), log(r)};
  return interp2AbsSpectralCoeff_->value(indepVars);
}

double
ParticleRadCoeffs::scattering_spectral_coeff( const double wavelength, const double r) const{
  double indepVars [] = {log(wavelength), log(r)};
  return interp2ScaSpectralCoeff_->value(indepVars);
}

double
ParticleRadCoeffs::planck_abs_coeff( const double r, const double T) const{
  double indepVars [] = {log(r),  T };
  return interp2AbsPlanckCoeff_->value(indepVars);
}

double
ParticleRadCoeffs::planck_sca_coeff( const double r, const double T) const{
  double indepVars [] = {log(r),  T };
  return interp2ScaPlanckCoeff_->value(indepVars);
}

double
ParticleRadCoeffs::ross_abs_coeff( const double r, const double T) const{
  double indepVars [] = {log(r),  T };
  return interp2AbsRossCoeff_->value(indepVars);
}

double
ParticleRadCoeffs::ross_sca_coeff( const double r, const double T) const{
  double indepVars [] = {log(r),  T };
  return interp2ScaRossCoeff_->value(indepVars);
}

double
soot_abs_coeff( const double lambda,
               const double volFrac,
               const complex<double> refIndex )
{
  // jcs we could significantly improve the efficiency here by calculating some temporaries and using those...
  return 36*M_PI*refIndex.real()*abs(refIndex.imag())*volFrac / (
   ( (refIndex.real()*refIndex.real()-abs(refIndex.imag())*abs(refIndex.imag())+2 ) *
     (refIndex.real()*refIndex.real()-abs(refIndex.imag())*abs(refIndex.imag())+2 ) + 4*
     refIndex.real()*abs(refIndex.imag())*refIndex.real()*abs(refIndex.imag()) ) * lambda );
}

//==============================================================================

