/*
* The MIT License
*
* Copyright (c) 1997-2019 The University of Utah
*
is hereby granted, free of charge, to any person obtaining a copy
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

#include <CCA/Components/Arches/Utility/ForcingTurbulence.h>
#include <CCA/Components/Arches/UPSHelper.h>
#include <CCA/Components/Arches/Utility/GridInfo.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <complex>

using namespace Uintah;
using namespace ArchesCore;
using namespace std;

// Constructor -----------------------------------------------------------------
ForcingTurbulence::ForcingTurbulence( std::string task_name, int matl_index ) : TaskInterface( task_name, matl_index ) {
}

// Destructor ------------------------------------------------------------------
ForcingTurbulence::~ForcingTurbulence() {
}

TaskAssignedExecutionSpace ForcingTurbulence::loadTaskComputeBCsFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ForcingTurbulence::loadTaskInitializeFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ForcingTurbulence::loadTaskEvalFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ForcingTurbulence::loadTaskTimestepInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

//--------------------------------------------------------------------------------------------------
TaskAssignedExecutionSpace ForcingTurbulence::loadTaskRestartInitFunctionPointers()
{
  return TaskAssignedExecutionSpace::NONE_EXECUTION_SPACE;
}

// Problem Setup ---------------------------------------------------------------
void ForcingTurbulence::problemSetup( ProblemSpecP& db ) {

  // Parse from UPS file
  m_uVel_name = parse_ups_for_role( UVELOCITY_ROLE, db, ArchesCore::default_uVel_name );
  m_vVel_name = parse_ups_for_role( VVELOCITY_ROLE, db, ArchesCore::default_vVel_name );
  m_wVel_name = parse_ups_for_role( WVELOCITY_ROLE, db, ArchesCore::default_wVel_name );

  m_density_name = parse_ups_for_role( DENSITY_ROLE, db, "density" );

  const ProblemSpecP db_root = db->getRootNode();
  ProblemSpecP db_grid = db_root->findBlock("Grid")->findBlock("Level")->findBlock("Box");
  db_grid->require("resolution", m_gridRes);

  Nx = m_gridRes[0];  Ny = m_gridRes[1];  Nz = m_gridRes[2];
  Nt = Nx * Ny * Nz;

  m_Nbins = std::min(Nx,std::min(Ny,Nz));

}

// Create Local Labels ---------------------------------------------------------
void ForcingTurbulence::create_local_labels() {
}

// Initialization --------------------------------------------------------------
void ForcingTurbulence::register_initialize( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const bool packed_tasks ) {

  typedef ArchesFieldContainer AFC;

  register_variable( m_density_name    , AFC::REQUIRES, 0, AFC::NEWDW, variable_registry);
  register_variable( m_uVel_name       , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, m_task_name );
  register_variable( m_vVel_name       , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, m_task_name );
  register_variable( m_wVel_name       , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, m_task_name );
  register_variable( default_uMom_name , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, m_task_name );
  register_variable( default_vMom_name , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, m_task_name );
  register_variable( default_wMom_name , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, m_task_name );

}

void ForcingTurbulence::initialize( const Patch* patch, ArchesTaskInfoManager* tsk_info ) {

  compute_TKE( patch, tsk_info, TKE_spectrum_nm1 );

  double binSum = 0.0;
  // for( auto const& [key, val] : TKE_spectrum_nm1 ) binSum += val;
  for ( std::map<int, double>::iterator i = TKE_spectrum_nm1.begin(); i != TKE_spectrum_nm1.end(); i++) { binSum += i->second ; }

}

// Timestep Eval ---------------------------------------------------------------
void ForcingTurbulence::register_timestep_eval( std::vector<ArchesFieldContainer::VariableInformation>& variable_registry, const int time_substep , const bool packed_tasks) {

  typedef ArchesFieldContainer AFC;

  register_variable( m_density_name    , AFC::REQUIRES, 0, AFC::NEWDW, variable_registry, time_substep);
  register_variable( m_uVel_name       , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_vVel_name       , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( m_wVel_name       , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( default_uMom_name , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( default_vMom_name , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );
  register_variable( default_wMom_name , AFC::MODIFIES, 0, AFC::NEWDW, variable_registry, time_substep, m_task_name );

}

void ForcingTurbulence::eval( const Patch* patch, ArchesTaskInfoManager* tsk_info ) {

  eval_scale_TKE( patch, tsk_info );

}

// Definitions -----------------------------------------------------------------

// Compute spectral radii ------------------------------------------------------
int ForcingTurbulence::compute_iShell( int i, int j, int k, int Nx, int Ny, int Nz) {

  int K_shell;
  float Kx_max, Ky_max, Kz_max;

  Kx_max = Nx/2;     Ky_max = Ny/2;     Kz_max = Nz/2;
  if (i > Kx_max)  i = i - Nx;
  if (j > Ky_max)  j = j - Ny;
  if (k > Kz_max)  k = k - Nz;

  float rK = sqrt(i*i + j*j + k*k);
  rK       = floor(rK + 0.5);         // Spherical shell based on spectral radii
  K_shell  = static_cast<int>(rK);    // Cast to int

  return K_shell;                     // int

}

// Compute TKE spectra ---------------------------------------------------------
void ForcingTurbulence::compute_TKE( const Patch* patch, ArchesTaskInfoManager* tsk_info, std::map<int, double> &TKE_spectrum ) {

  // Domain size. Patch size excluding the ghost cells
  const Level* lvl = patch->getLevel();
  IntVector min; IntVector max;
  lvl->getGrid()->getLevel(0)->findCellIndexRange(min,max);
  IntVector period_bc = IntVector(1,1,1) - lvl->getPeriodicBoundaries();

/*  Unused variables  -Todd
  Box domainBox = lvl->getBox(min+period_bc, max-period_bc);
  const double lowx = domainBox.lower().x();
  const double lowy = domainBox.lower().y();
  const double lowz = domainBox.lower().z();

  const double upx = domainBox.upper().x();
  const double upy = domainBox.upper().y();
  const double upz = domainBox.upper().z();

  const double Lx = upx - lowx;
  const double Ly = upy - lowy;
  const double Lz = upz - lowz;
*/
 
  // Get the velocity field
  SFCXVariable<double>& uVel  = tsk_info->get_field<SFCXVariable<double> >(m_uVel_name);
  SFCYVariable<double>& vVel  = tsk_info->get_field<SFCYVariable<double> >(m_vVel_name);
  SFCZVariable<double>& wVel  = tsk_info->get_field<SFCZVariable<double> >(m_wVel_name);

  constCCVariable<double>& Rho = tsk_info->get_field<constCCVariable<double> >(m_density_name);
  SFCXVariable<double>& uMom   = tsk_info->get_field<SFCXVariable<double> >(default_uMom_name);
  SFCYVariable<double>& vMom   = tsk_info->get_field<SFCYVariable<double> >(default_vMom_name);
  SFCZVariable<double>& wMom   = tsk_info->get_field<SFCZVariable<double> >(default_wMom_name);

  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    int i = c[0];    int j = c[1];   int k = c[2];

    uVel(i,j,k) = uMom(i,j,k) / Rho(i,j,k) ;  // Velocity from momentum term
    vVel(i,j,k) = vMom(i,j,k) / Rho(i,j,k) ;
    wVel(i,j,k) = wMom(i,j,k) / Rho(i,j,k) ;

  };

  // Spectral fields
  std::complex<double> TKEh[Nx*Ny*Nz];
  std::complex<double> Uh[Nx*Ny*Nz];
  std::complex<double> Vh[Nx*Ny*Nz];
  std::complex<double> Wh[Nx*Ny*Nz];

  // Transform raw velocities into spectral space
  fft3D( patch, Nx, Ny, Nz, 1, uVel, Uh, vVel, Vh, wVel, Wh );

  // double totalKE = 0.0;
  // Compute TKE and Integrate over the spectral shell
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    int i = c[0];    int j = c[1];   int k = c[2];

    // Locate the i,j,k in spectral space
    int K_shell = compute_iShell(i, j, k, Nx, Ny, Nz);
    int locate = k + Ny*j + Nz*Ny*i;

    // Compute KE
    TKEh[locate] = 0.5*(Uh[locate] * std::conj(Uh[locate]) +
                        Vh[locate] * std::conj(Vh[locate]) +
                        Wh[locate] * std::conj(Wh[locate]) );
    // Bin the KE
    auto wave_ptr = TKE_spectrum.find(K_shell);   int waveNo = wave_ptr->first;   double binTKE = wave_ptr->second;
    binTKE = binTKE + TKEh[locate].real();                              // Append the binned energy
    TKE_spectrum[waveNo] = binTKE;                                      // Update the binned energy

    // totalKE = totalKE + TKEh[locate].real();                            // System TKE

  };

}

// Scale TKE -------------------------------------------------------------------
void ForcingTurbulence::eval_scale_TKE( const Patch* patch, ArchesTaskInfoManager* tsk_info ) {

  // Domain size. Patch size excluding the ghost cells
  const Level* lvl = patch->getLevel();
  IntVector min; IntVector max;
  lvl->getGrid()->getLevel(0)->findCellIndexRange(min,max);
  IntVector period_bc = IntVector(1,1,1) - lvl->getPeriodicBoundaries();

/*  unused variables --Todd
  Box domainBox = lvl->getBox(min+period_bc, max-period_bc);
  const double lowx = domainBox.lower().x();
  const double lowy = domainBox.lower().y();
  const double lowz = domainBox.lower().z();

  const double upx = domainBox.upper().x();
  const double upy = domainBox.upper().y();
  const double upz = domainBox.upper().z();

  const double Lx = upx - lowx;
  const double Ly = upy - lowy;
  const double Lz = upz - lowz;
*/
  // Get the velocity field
  SFCXVariable<double>& uVel  = tsk_info->get_field<SFCXVariable<double> >(m_uVel_name);
  SFCYVariable<double>& vVel  = tsk_info->get_field<SFCYVariable<double> >(m_vVel_name);
  SFCZVariable<double>& wVel  = tsk_info->get_field<SFCZVariable<double> >(m_wVel_name);

  constCCVariable<double>& Rho = tsk_info->get_field<constCCVariable<double> >(m_density_name);
  SFCXVariable<double>& uMom   = tsk_info->get_field<SFCXVariable<double> >(default_uMom_name);
  SFCYVariable<double>& vMom   = tsk_info->get_field<SFCYVariable<double> >(default_vMom_name);
  SFCZVariable<double>& wMom   = tsk_info->get_field<SFCZVariable<double> >(default_wMom_name);

  // Compute the TKE at current state
  // for( auto const& [key, val] : TKE_spectrum_n ) { TKE_spectrum_n[key] = 0.0; }
  for ( std::map<int, double>::iterator i = TKE_spectrum_n.begin(); i != TKE_spectrum_n.end(); i++) { i->second = 0.0 ; }
  compute_TKE( patch, tsk_info, TKE_spectrum_n );

  // Spectral fields
  std::complex<double> TKEh[Nx*Ny*Nz];
  std::complex<double> Uh[Nx*Ny*Nz];
  std::complex<double> Vh[Nx*Ny*Nz];
  std::complex<double> Wh[Nx*Ny*Nz];

  // Transform raw velocities into spectral space
  fft3D( patch, Nx, Ny, Nz, 1, uVel, Uh, vVel, Vh, wVel, Wh );

  // ---------------------------------------------------------------------------
  std::map<int, double> TKE_spectrum_scaled;
  double totalKE = 0.0;
  // Compute TKE and Integrate over the spectral shell
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    int i = c[0];    int j = c[1];   int k = c[2];

    // Locate the i,j,k in spectral space
    int K_shell = compute_iShell(i, j, k, Nx, Ny, Nz);
    int locate = k + Ny*j + Nz*Ny*i;

    //Compute the scaling factor for the current i,j,k
    std::complex<double> scaleE(0.0, 0.0);
    auto Spec_nm1_K = TKE_spectrum_nm1.find(K_shell);     double TKE_nm1_K = Spec_nm1_K->second;
    auto Spec_n_K   = TKE_spectrum_n.find(K_shell);       double TKE_n_K   = Spec_n_K->second;
    scaleE.real( sqrt( TKE_nm1_K / TKE_n_K ) );

    // Scale the spectral velocity fields
    if ( K_shell <= 3 ) {
      Uh[locate] = scaleE * Uh[locate];
      Vh[locate] = scaleE * Vh[locate];
      Wh[locate] = scaleE * Wh[locate];
    }

    // Compute the scaled TKE
    TKEh[locate] = 0.5*( Uh[locate] * std::conj(Uh[locate]) +
                         Vh[locate] * std::conj(Vh[locate]) +
                         Wh[locate] * std::conj(Wh[locate]) );
    // Bin the KE
    auto wave_ptr = TKE_spectrum_scaled.find(K_shell);   int waveNo = wave_ptr->first;   double binTKE = wave_ptr->second;
    binTKE = binTKE + TKEh[locate].real();                              // Append the binned energy
    TKE_spectrum_scaled[waveNo] = binTKE;                               // Update the binned energy

    totalKE = totalKE + TKEh[locate].real();                            // System TKE

  };

  // Store the new scaled TKE for next timestep
  // for( auto const& [key, val] : TKE_spectrum_nm1 ) TKE_spectrum_nm1[key] = TKE_spectrum_scaled[key];
  TKE_spectrum_nm1 = TKE_spectrum_scaled;

  // Transform the scaled spectral velocitites into real space
  fft3D( patch, Nx, Ny, Nz, -1, uVel, Uh, vVel, Vh, wVel, Wh );

  // ---------------------------------------------------------------------------
  // Update the momentum field
  for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
    IntVector c = *iter;
    int i = c[0];    int j = c[1];   int k = c[2];

    uMom(i,j,k)  = Rho(i,j,k)  * uVel(i,j,k) ;
    vMom(i,j,k)  = Rho(i,j,k)  * vVel(i,j,k) ;
    wMom(i,j,k)  = Rho(i,j,k)  * wVel(i,j,k) ;

  };

  double binSum = 0.0;
  // for( auto const& [key, val] : TKE_spectrum_scaled ) binSum += val;
  for ( std::map<int, double>::iterator i = TKE_spectrum_scaled.begin(); i != TKE_spectrum_scaled.end(); i++) { binSum += i->second ; }

}

// Compute FFT -----------------------------------------------------------------

/* =============================================================================
  Adapted from Dr. Wang Jian-Sheng's FFT C++ implementation.
  https://www.physics.nus.edu.sg/~phywjs/
  https://www.physics.nus.edu.sg/~phywjs/CZ5101/fft.c

  References :

  Computational Frameworks for the Fast Fourier Transform (1992)
    Frontiers in Applied Mathematics, Charles Van Loan,
    Cornell University, Ithaca, New York

  Cooley and Tukey, An algorithm for the machine calculation of complex
    Fourier series, Math. of Computation, vol. 19, pp. 297–301, 1965

============================================================================= */

void ForcingTurbulence::fft3D( const Patch* patch, int n1, int n2, int n3, int flag,
                               SFCXVariable<double>& Uvel_Real, std::complex<double> *Uvel_Spectral,
                               SFCYVariable<double>& Vvel_Real, std::complex<double> *Vvel_Spectral,
                               SFCZVariable<double>& Wvel_Real, std::complex<double> *Wvel_Spectral ) {

  assert(1 == flag || -1 == flag);

  // ---------------------------------------------------------------------------
  if(flag == 1 ) {                                          // Forward transform

    // Create a temp flattened spectral field with real velocities
    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      int i = c[0];
      int j = c[1];
      int k = c[2];

      int locate = k + n2*j + n3*n2*i;

      Uvel_Spectral[locate].real(Uvel_Real(i,j,k));
      Uvel_Spectral[locate].imag(0.0);

      Vvel_Spectral[locate].real(Vvel_Real(i,j,k));
      Vvel_Spectral[locate].imag(0.0);

      Wvel_Spectral[locate].real(Wvel_Real(i,j,k));
      Wvel_Spectral[locate].imag(0.0);
    }

    int i, n, n23;
    n23 = n2*n3;
    n = n1*n23;

    std::complex<double> *y;
    y = (std::complex<double> *) malloc( n23*sizeof(std::complex<double>) ); assert(NULL != y);
    for(i=0; i < n; i += n3)  { stockham(Uvel_Spectral+i, y, n3,   1, flag); }    // Z
    for(i=0; i < n; i += n23) { stockham(Uvel_Spectral+i, y, n23, n3, flag); }    // Y
    free(y); cooley_tukey(Uvel_Spectral, n, n23, flag);                           // X

    // std::complex<double> *y;
    y = (std::complex<double> *) malloc( n23*sizeof(std::complex<double>) ); assert(NULL != y);
    for(i=0; i < n; i += n3)  { stockham(Vvel_Spectral+i, y, n3,   1, flag); }    // Z
    for(i=0; i < n; i += n23) { stockham(Vvel_Spectral+i, y, n23, n3, flag); }    // Y
    free(y); cooley_tukey(Vvel_Spectral, n, n23, flag);                           // X

    // std::complex<double> *y;
    y = (std::complex<double> *) malloc( n23*sizeof(std::complex<double>) ); assert(NULL != y);
    for(i=0; i < n; i += n3)  { stockham(Wvel_Spectral+i, y, n3,   1, flag); }    // Z
    for(i=0; i < n; i += n23) { stockham(Wvel_Spectral+i, y, n23, n3, flag); }    // Y
    free(y); cooley_tukey(Wvel_Spectral, n, n23, flag);                           // X

  }

  // ---------------------------------------------------------------------------
  else if (flag == -1 ) {                                   // Inverse transform

    int i, n, n23;
    n23 = n2*n3;
    n = n1*n23;

    std::complex<double> *y;
    y = (std::complex<double> *) malloc( n23*sizeof(std::complex<double>) ); assert(NULL != y);
    for(i=0; i < n; i += n3)  { stockham(Uvel_Spectral+i, y, n3,   1, flag); }    // Z
    for(i=0; i < n; i += n23) { stockham(Uvel_Spectral+i, y, n23, n3, flag); }    // Y
    free(y); cooley_tukey(Uvel_Spectral, n, n23, flag);                           // X

    // std::complex<double> *y;
    y = (std::complex<double> *) malloc( n23*sizeof(std::complex<double>) ); assert(NULL != y);
    for(i=0; i < n; i += n3)  { stockham(Vvel_Spectral+i, y, n3,   1, flag); }    // Z
    for(i=0; i < n; i += n23) { stockham(Vvel_Spectral+i, y, n23, n3, flag); }    // Y
    free(y); cooley_tukey(Vvel_Spectral, n, n23, flag);                           // X

    // std::complex<double> *y;
    y = (std::complex<double> *) malloc( n23*sizeof(std::complex<double>) ); assert(NULL != y);
    for(i=0; i < n; i += n3)  { stockham(Wvel_Spectral+i, y, n3,   1, flag); }    // Z
    for(i=0; i < n; i += n23) { stockham(Wvel_Spectral+i, y, n23, n3, flag); }    // Y
    free(y); cooley_tukey(Wvel_Spectral, n, n23, flag);                           // X

    // Assign the inverse flattened transformation to the Real value field container
    for(CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector c = *iter;
      int i = c[0];
      int j = c[1];
      int k = c[2];

      int locate = k + n2*j + n3*n2*i;

      Uvel_Real(i,j,k) = std::real(Uvel_Spectral[locate])/n;
      Vvel_Real(i,j,k) = std::real(Vvel_Spectral[locate])/n;
      Wvel_Real(i,j,k) = std::real(Wvel_Spectral[locate])/n;
    }
  }
}

// @jelias
void ForcingTurbulence::stockham( std::complex<double> x[], std::complex<double> y[], int nt, int n2, int flag ) {

  std::complex<double> *y_src, *tmp;
  int i, j, k;
  int Lstar, r, j_rSq, k2;
  int half, m1, m2;
  double wr, wi, tr, ti;

  y_src = y;
  r = half = nt >> 1;
  Lstar = 1;

  while(r >= n2) {                                          // log2(nt/n2)
    tmp = x;     x = y;     y = tmp;                        // Swap pointers : x = new, y = old
    m1 = 0;                                                 // First half
    m2 = half;                                              // Second half
    for(j = 0; j < Lstar; ++j) {
      wr = cos(M_PI*j/Lstar);                               // Re & Im
      wi = -flag * sin(M_PI*j/Lstar);
      j_rSq = j*(r+r);
      for(k = j_rSq; k < j_rSq+r; ++k) {                    // Butterfly operation
        k2 = k + r;
        tr =  wr*std::real(y[k2]) - wi*std::imag(y[k2]);
        ti =  wr*std::imag(y[k2]) + wi*std::real(y[k2]);
        x[m1].real(std::real(y[k]) + tr);
        x[m1].imag(std::imag(y[k]) + ti);
        x[m2].real(std::real(y[k]) - tr);
        x[m2].imag(std::imag(y[k]) - ti);
        ++m1;
        ++m2;
      }
    }
    r     >>= 1;
    Lstar <<= 1;
  };

  if (y != y_src) {
    for(i = 0; i < nt; ++i) {
      y[i] = x[i];
    }
  }
  assert(Lstar == nt/n2);
  assert(1 == nt || m2 == nt);
}

void ForcingTurbulence::cooley_tukey( std::complex<double> x[], int nt, int n2, int flag ) {

  std::complex<double> c;
  int i, j, k, m, p, n1;
  int Ls, ks, ms, jm, dk;
  double wr, wi, tr, ti;

  n1 = nt/n2;                                         // Bit reversal permutation
  for(k = 0; k < n1; ++k) {
    j = 0;
    m = k;
    p = 1;
    while(p < n1) {
      j = 2*j + (m&1);
      m >>= 1;
      p <<= 1;
    }
    assert(p == n1);
    if(j > k) {
      for(i = 0; i < n2; ++i) {
        c = x[k*n2+i];
        x[k*n2+i] = x[j*n2+i];
        x[j*n2+i] = c;
      }
    }
  }

  p = 1;
  while(p < n1) {
    Ls = p;
    p <<= 1;
    jm = 0;
    dk = p*n2;
    for(j = 0; j < Ls; ++j) {
      wr = cos(M_PI * j / Ls);                                // Re & Im
      wi = -flag * sin(M_PI*j/Ls);
      for(k = jm; k < nt; k += dk) {                          // Butterfly operation
        ks = k + Ls*n2;
        for(i = 0; i < n2; ++i) {
          m = k + i;
          ms = ks + i;
          tr =  wr*std::real(x[ms]) - wi*std::imag(x[ms]);
          ti =  wr*std::imag(x[ms]) + wi*std::real(x[ms]);
          x[ms].real(std::real(x[m]) - tr);
          x[ms].imag(std::imag(x[m]) - ti);
          x[m].real(std::real(x[m]) + tr);
          x[m].imag(std::imag(x[m]) + ti);
        }
      }
    jm += n2;
    }
  }
}
