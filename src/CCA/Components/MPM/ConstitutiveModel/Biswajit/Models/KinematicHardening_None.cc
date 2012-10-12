/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifdef __APPLE__
// This is a hack.  gcc 3.3 #undefs isnan in the cmath header, which
// make the isnan function not work.  This define makes the cmath header
// not get included since we do not need it anyway.
#  define _CPP_CMATH
#endif

#include <CCA/Components/MPM/ConstitutiveModel/Biswajit/Models/KinematicHardening_None.h>
#include <cmath>


using namespace Uintah;
using namespace UintahBB;

KinematicHardening_None::KinematicHardening_None()
{
}
         
KinematicHardening_None::KinematicHardening_None(ProblemSpecP& ps)
{
}
         
KinematicHardening_None::KinematicHardening_None(const KinematicHardening_None* cm)
{
}
         
KinematicHardening_None::~KinematicHardening_None()
{
}

void KinematicHardening_None::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP plastic_ps = ps->appendChild("kinematic_hardening_model");
  plastic_ps->setAttribute("type","none");
}

void 
KinematicHardening_None::computeBackStress(const ModelState* state,
                                        const double& delT,
                                        const particleIndex idx,
                                        const double& delLambda,
                                        const Matrix3& df_dsigma_new,
                                        const Matrix3& backStress_old,
                                        Matrix3& backStress_new)
{
  Matrix3 Zero(0.0);
  backStress_new = Zero;
  return;
}


void 
KinematicHardening_None::eval_h_beta(const Matrix3& df_dsigma,
                                  const ModelState* ,
                                  Matrix3& h_beta)
{
  Matrix3 Zero(0.0);
  h_beta = Zero;
  return;
}


