/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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


#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/Arches.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Patch.h>
#include <Core/Geometry/Point.h>
#include <Core/Exceptions/InvalidValue.h>
#include <Core/Exceptions/ParameterNotFound.h>
#include <iostream>

using namespace std;
using namespace Uintah;

#include <CCA/Components/Arches/fortran/cellg_fort.h>

CellInformation::CellInformation(const Patch* patch)
{
  IntVector domLo = patch->getExtraCellLowIndex(Arches::ONEGHOSTCELL);
  IntVector domHi = patch->getExtraCellHighIndex(Arches::ONEGHOSTCELL);

  IntVector locationLo = patch->getExtraCellLowIndex(Arches::THREEGHOSTCELLS);
  IntVector locationHi = patch->getExtraCellHighIndex(Arches::THREEGHOSTCELLS);

  IntVector idxLoU = patch->getSFCXFORTLowIndex__Old();
  IntVector idxHiU = patch->getSFCXFORTHighIndex__Old();
  IntVector idxLoV = patch->getSFCYFORTLowIndex__Old();
  IntVector idxHiV = patch->getSFCYFORTHighIndex__Old();
  IntVector idxLoW = patch->getSFCZFORTLowIndex__Old();
  IntVector idxHiW = patch->getSFCZFORTHighIndex__Old();
 
  IntVector idxLo = patch->getFortranCellLowIndex();
  IntVector idxHi = patch->getFortranCellHighIndex();

  // grid cell information
  xx.resize(locationLo.x(), locationHi.x());
  yy.resize(locationLo.y(), locationHi.y());
  zz.resize(locationLo.z(), locationHi.z());

  const Level* level = patch->getLevel();
  for (int ii = locationLo.x(); ii < locationHi.x(); ii++){
    xx[ii] = level->getCellPosition(IntVector(ii, locationLo.y(), locationLo.z())).x();
  }
  for (int ii = locationLo.y(); ii < locationHi.y(); ii++){
    yy[ii] = level->getCellPosition(IntVector(locationLo.x(), ii, locationLo.z())).y();
  }
  for (int ii = locationLo.z(); ii < locationHi.z(); ii++){
    zz[ii] = level->getCellPosition(IntVector(locationLo.x(), locationLo.y(), ii)).z();
  }

  bool xminus = patch->getBCType(Patch::xminus) != Patch::Neighbor;
  bool xplus =  patch->getBCType(Patch::xplus) != Patch::Neighbor;
  bool yminus = patch->getBCType(Patch::yminus) != Patch::Neighbor;
  bool yplus =  patch->getBCType(Patch::yplus) != Patch::Neighbor;
  bool zminus = patch->getBCType(Patch::zminus) != Patch::Neighbor;
  bool zplus =  patch->getBCType(Patch::zplus) != Patch::Neighbor;

/*  int Nx = 12;
  double alpha_x = 0.5;
  double l_x = 3.0;
  int ist = locationLo.x();
  int iend = locationHi.x();
  if (xminus) ist ++;
  if (xplus) iend --;
  for (int ii = ist; ii < iend; ii++)
    xx[ii] = (alpha_x*l_x*(ii+1)/Nx + ii*(ii+1)*(1.0-alpha_x)*l_x/((Nx-1)*Nx) + (alpha_x*l_x*ii/Nx + ii*(ii-1)*(1.0-alpha_x)*l_x/((Nx-1)*Nx)))/2.0;
 if (xminus) xx[-1] = -xx[0];
 if (xplus) xx[Nx] = l_x + l_x - xx[Nx-1];
  for (int ii = locationLo.x(); ii < locationHi.x(); ii++)
  cout << xx[ii] << endl;



  double alpha_y, l_y;
  double l =3.0;
  int ist_y, iend_y;
  int N=12;
  int Ny=N/4;
  int start=locationLo.y();
  int end=locationHi.y();
  l_y = l/3.0;
  if (yminus) {
  alpha_y = 1.5;
  ist_y = locationLo.y();
  ist_y ++;
  iend_y = ist_y+Ny;
  for (int ii = ist_y; ii < iend_y; ii++) 
    yy[ii] = (alpha_y*l_y*(ii+1)/Ny + ii*(ii+1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny) + (alpha_y*l_y*ii/Ny + ii*(ii-1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny)))/2.0;
  
  yy[-1] = -yy[0];
  start=iend_y;
  }
  if (yplus) {
  alpha_y = 0.5;
  iend_y = locationHi.y();
  iend_y --;
  ist_y = iend_y-Ny;
  for (int jj = ist_y; jj < iend_y; jj++) {
    int ii=jj-ist_y;
    yy[jj] = 2.0+(alpha_y*l_y*(ii+1)/Ny + ii*(ii+1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny) + (alpha_y*l_y*ii/Ny + ii*(ii-1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny)))/2.0;
  }
  yy[N] = l + l - yy[N-1];
  end=ist_y;
  }
  for (int ii = start; ii < end; ii++)
    yy[ii] =1.0+1.0/(N/2)/2.0+ 1.0/(N/2)*(ii-start);
  if (yplus) {
  for (int ii = start; ii < end; ii++)
    yy[ii] =2.0+1.0/(N/2)/2.0- 1.0/(N/2)*(end-ii);
  }
  for (int ii = locationLo.y(); ii < locationHi.y(); ii++)
  cout << ii<< " y " <<yy[ii] << endl;

  start=locationLo.z();
  end=locationHi.z();
  if (zminus) {
  alpha_y = 1.5;
  ist_y = locationLo.z();
  ist_y ++;
  iend_y = ist_y+Ny;
  for (int ii = ist_y; ii < iend_y; ii++)
    zz[ii] = (alpha_y*l_y*(ii+1)/Ny + ii*(ii+1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny) + (alpha_y*l_y*ii/Ny + ii*(ii-1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny)))/2.0;
  zz[-1] = -zz[0];
  start=iend_y;
  }
  if (zplus) {
  alpha_y = 0.5;
  iend_y = locationHi.z();
  iend_y --;
  ist_y = iend_y-Ny;
  for (int jj = ist_y; jj < iend_y; jj++) {
    int ii=jj-ist_y;
    zz[jj] = 2.0+(alpha_y*l_y*(ii+1)/Ny + ii*(ii+1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny) + (alpha_y*l_y*ii/Ny + ii*(ii-1)*(1.0-alpha_y)*l_y/((Ny-1)*Ny)))/2.0;
  }
  zz[N] = l + l - zz[N-1];
  end=ist_y;
  }
  for (int ii = start; ii < end; ii++)
    zz[ii] =1.0+1.0/(N/2)/2.0+ 1.0/(N/2)*(ii-start);
  if (zplus) {
  for (int ii = start; ii < end; ii++)
    zz[ii] =2.0+1.0/(N/2)/2.0- 1.0/(N/2)*(end-ii);
  }
  for (int ii = locationLo.z(); ii < locationHi.z(); ii++)
  cout << ii<< " z " <<zz[ii] << endl;
*/

  //  allocate memory for x-dim arrays
  dxep.resize(domLo.x(), domHi.x());
  dxpw.resize(domLo.x(), domHi.x());
  sew.resize(domLo.x(), domHi.x());
  xu.resize(domLo.x(), domHi.x());
  dxpwu.resize(domLo.x(), domHi.x());
  dxepu.resize(domLo.x(), domHi.x());
  sewu.resize(domLo.x(), domHi.x());
  cee.resize(domLo.x(), domHi.x());
  cww.resize(domLo.x(), domHi.x());
  cwe.resize(domLo.x(), domHi.x());
  ceeu.resize(domLo.x(), domHi.x());
  cwwu.resize(domLo.x(), domHi.x());
  cweu.resize(domLo.x(), domHi.x());
  efac.resize(domLo.x(), domHi.x());
  wfac.resize(domLo.x(), domHi.x());
  fac1u.resize(domLo.x(), domHi.x());
  fac2u.resize(domLo.x(), domHi.x());
  iesdu.resize(domLo.x(), domHi.x());
  fac3u.resize(domLo.x(), domHi.x());
  fac4u.resize(domLo.x(), domHi.x());
  iwsdu.resize(domLo.x(), domHi.x());
  fac1ew.resize(domLo.x(), domHi.x());
  fac2ew.resize(domLo.x(), domHi.x());
  e_shift.resize(domLo.x(), domHi.x());
  fac3ew.resize(domLo.x(), domHi.x());
  fac4ew.resize(domLo.x(), domHi.x());
  w_shift.resize(domLo.x(), domHi.x());
  // allocate memory for y-dim arrays
  dynp.resize(domLo.y(), domHi.y());
  dyps.resize(domLo.y(), domHi.y());
  sns.resize(domLo.y(), domHi.y());
  yv.resize(domLo.y(), domHi.y());
  dynpv.resize(domLo.y(), domHi.y());
  dypsv.resize(domLo.y(), domHi.y());
  snsv.resize(domLo.y(), domHi.y());
  cnn.resize(domLo.y(), domHi.y());
  css.resize(domLo.y(), domHi.y());
  csn.resize(domLo.y(), domHi.y());
  cnnv.resize(domLo.y(), domHi.y());
  cssv.resize(domLo.y(), domHi.y());
  csnv.resize(domLo.y(), domHi.y());
  nfac.resize(domLo.y(), domHi.y());
  sfac.resize(domLo.y(), domHi.y());
  fac1v.resize(domLo.y(), domHi.y());
  fac2v.resize(domLo.y(), domHi.y());
  jnsdv.resize(domLo.y(), domHi.y());
  fac3v.resize(domLo.y(), domHi.y());
  fac4v.resize(domLo.y(), domHi.y());
  jssdv.resize(domLo.y(), domHi.y());
  fac1ns.resize(domLo.y(), domHi.y());
  fac2ns.resize(domLo.y(), domHi.y());
  n_shift.resize(domLo.y(), domHi.y());
  fac3ns.resize(domLo.y(), domHi.y());
  fac4ns.resize(domLo.y(), domHi.y());
  s_shift.resize(domLo.y(), domHi.y());
  //allocate memory for z-dim arrays
  dztp.resize(domLo.z(), domHi.z());
  dzpb.resize(domLo.z(), domHi.z());
  stb.resize(domLo.z(), domHi.z());
  zw.resize(domLo.z(), domHi.z());
  dztpw.resize(domLo.z(), domHi.z());
  dzpbw.resize(domLo.z(), domHi.z());
  stbw.resize(domLo.z(), domHi.z());
  ctt.resize(domLo.z(), domHi.z());
  cbb.resize(domLo.z(), domHi.z());
  cbt.resize(domLo.z(), domHi.z());
  cttw.resize(domLo.z(), domHi.z());
  cbbw.resize(domLo.z(), domHi.z());
  cbtw.resize(domLo.z(), domHi.z());
  tfac.resize(domLo.z(), domHi.z());
  bfac.resize(domLo.z(), domHi.z());
  fac1w.resize(domLo.z(), domHi.z());
  fac2w.resize(domLo.z(), domHi.z());
  ktsdw.resize(domLo.z(), domHi.z());
  fac3w.resize(domLo.z(), domHi.z());
  fac4w.resize(domLo.z(), domHi.z());
  kbsdw.resize(domLo.z(), domHi.z());
  fac1tb.resize(domLo.z(), domHi.z());
  fac2tb.resize(domLo.z(), domHi.z());
  t_shift.resize(domLo.z(), domHi.z());
  fac3tb.resize(domLo.z(), domHi.z());
  fac4tb.resize(domLo.z(), domHi.z());
  b_shift.resize(domLo.z(), domHi.z());
  // for fortran
  domHi = domHi - IntVector(1,1,1);


  // for computing geometry parameters
  fort_cellg(domLo, domHi, idxLoU, idxHiU, idxLoV, idxHiV,
             idxLoW, idxHiW, idxLo, idxHi,
             sew, sns, stb, sewu, snsv, stbw, dxep, dynp, dztp,
             dxepu, dynpv, dztpw, dxpw, dyps, dzpb, dxpwu, dypsv, dzpbw,
             cee, cwe, cww, ceeu, cweu, cwwu, cnn, csn, css,
             cnnv, csnv, cssv, ctt, cbt, cbb, cttw, cbtw, cbbw,
             xx, xu, yy, yv, zz, zw, efac, wfac, nfac, sfac, tfac, bfac,
             fac1u, fac2u, fac3u, fac4u, fac1v, fac2v, fac3v, fac4v,
             fac1w, fac2w, fac3w, fac4w, iesdu, iwsdu, jnsdv, jssdv, 
             ktsdw, kbsdw, fac1ew, fac2ew, fac3ew, fac4ew,
             fac1ns, fac2ns, fac3ns, fac4ns, fac1tb, fac2tb, fac3tb, fac4tb,
             e_shift, w_shift, n_shift, s_shift, t_shift, b_shift,
             xminus,xplus,yminus,yplus,zminus,zplus);

}

CellInformation::~CellInformation()
{
}

