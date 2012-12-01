/*

 The MIT License

 Copyright (c) 1997-2012 The University of Utah

 License for the specific language governing rights and limitations under
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.

 */

#include <CCA/Components/MD/MD.h>
#include <CCA/Components/MD/SPMEGrid.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/SimpleMaterial.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/ParticleVariable.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>

using namespace Uintah;

static DebugStream md_dbg("MDDebug", false);

MD::MD(const ProcessorGroup* myworld) :
    UintahParallelComponent(myworld)
{
  lb = scinew MDLabel();
}

MD::~MD()
{
  delete lb;
}

void MD::problemSetup(const ProblemSpecP& params,
                      const ProblemSpecP& restart_prob_spec,
                      GridP& /*grid*/,
                      SimulationStateP& sharedState)
{
  d_sharedState_ = sharedState;
  dynamic_cast<Scheduler*>(getPort("scheduler"))->setPositionVar(lb->pXLabel);
  ProblemSpecP ps = params->findBlock("MD");

  ps->get("coordinateFile", coordinateFile_);
  ps->get("numAtoms", numAtoms_);
  ps->get("boxSize", box_);
  ps->get("cutoffRadius", cutoffRadius_);
  ps->get("R12", R12_);
  ps->get("R6", R6_);

  mymat_ = scinew SimpleMaterial();
  d_sharedState_->registerSimpleMaterial(mymat_);

  // do file I/O to get atom coordinates and simulation cell size
  extractCoordinates();

  // for neighbor indices
  for (unsigned int i = 0; i < numAtoms_; i++) {
    neighborList.push_back(vector<int>(0));
  }

  // create neighbor list for each atom in the system
  generateNeighborList();
}

void MD::scheduleInitialize(const LevelP& level,
                            SchedulerP& sched)
{
  Task* task = scinew Task("initialize", this, &MD::initialize);
  task->computes(lb->pXLabel);
  task->computes(lb->pForceLabel);
  task->computes(lb->pAccelLabel);
  task->computes(lb->pVelocityLabel);
  task->computes(lb->pEnergyLabel);
  task->computes(lb->pMassLabel);
  task->computes(lb->pChargeLabel);
  task->computes(lb->pParticleIDLabel);
  task->computes(lb->vdwEnergyLabel);
  sched->addTask(task, level->eachPatch(), d_sharedState_->allMaterials());
}

void MD::scheduleComputeStableTimestep(const LevelP& level,
                                       SchedulerP& sched)
{
  Task* task = scinew Task("computeStableTimestep", this, &MD::computeStableTimestep);
  task->requires(Task::NewDW, lb->vdwEnergyLabel);
  task->computes(d_sharedState_->get_delt_label(), level.get_rep());
  sched->addTask(task, level->eachPatch(), d_sharedState_->allMaterials());
}

void MD::scheduleTimeAdvance(const LevelP& level,
                             SchedulerP& sched)
{
  const PatchSet* patches = level->eachPatch();
  const MaterialSet* matls = d_sharedState_->allMaterials();

  d_particleState.clear();
  d_particleState_preReloc.clear();

  d_particleState.resize(matls->size());
  d_particleState_preReloc.resize(matls->size());

  scheduleCalculateNonBondedForces(sched, patches, matls);
  scheduleUpdatePosition(sched, patches, matls);

  sched->scheduleParticleRelocation(level, lb->pXLabel_preReloc, d_particleState_preReloc, lb->pXLabel, d_particleState,
                                    lb->pParticleIDLabel, matls);
}

void MD::computeStableTimestep(const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* /*matls*/,
                               DataWarehouse*,
                               DataWarehouse* new_dw)
{
  if (pg->myrank() == 0) {
    sum_vartype vdwEnergy;
    new_dw->get(vdwEnergy, lb->vdwEnergyLabel);
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << "Total Energy = " << std::setprecision(16) << vdwEnergy << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;
    std::cout << std::endl;
  }
  new_dw->put(delt_vartype(1), d_sharedState_->get_delt_label(), getLevel(patches));
}

void MD::scheduleCalculateNonBondedForces(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls)
{
  Task* task = scinew Task("calculateNonBondedForces", this, &MD::calculateNonBondedForces);

  task->requires(Task::OldDW, lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pEnergyLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);

  task->computes(lb->pForceLabel_preReloc);
  task->computes(lb->pEnergyLabel_preReloc);
  task->computes(lb->vdwEnergyLabel);

  sched->addTask(task, patches, matls);

  // for particle relocation
  for (int m = 0; m < matls->size(); m++) {
    d_particleState_preReloc[m].push_back(lb->pForceLabel_preReloc);
    d_particleState_preReloc[m].push_back(lb->pEnergyLabel_preReloc);
    d_particleState[m].push_back(lb->pForceLabel);
    d_particleState[m].push_back(lb->pEnergyLabel);
  }
}

void MD::scheduleUpdatePosition(SchedulerP& sched,
                                const PatchSet* patches,
                                const MaterialSet* matls)
{
  Task* task = scinew Task("updatePosition", this, &MD::updatePosition);

  task->requires(Task::OldDW, lb->pXLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pForceLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pAccelLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pVelocityLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pMassLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pChargeLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, lb->pParticleIDLabel, Ghost::AroundNodes, SHRT_MAX);
  task->requires(Task::OldDW, d_sharedState_->get_delt_label());

  task->computes(lb->pXLabel_preReloc);
  task->computes(lb->pAccelLabel_preReloc);
  task->computes(lb->pVelocityLabel_preReloc);
  task->computes(lb->pMassLabel_preReloc);
  task->computes(lb->pChargeLabel_preReloc);
  task->computes(lb->pParticleIDLabel_preReloc);

  sched->addTask(task, patches, matls);

  // for particle relocation
  for (int m = 0; m < matls->size(); m++) {
    d_particleState_preReloc[m].push_back(lb->pAccelLabel_preReloc);
    d_particleState_preReloc[m].push_back(lb->pVelocityLabel_preReloc);
    d_particleState_preReloc[m].push_back(lb->pMassLabel_preReloc);
    d_particleState_preReloc[m].push_back(lb->pChargeLabel_preReloc);
    d_particleState_preReloc[m].push_back(lb->pParticleIDLabel_preReloc);
    d_particleState[m].push_back(lb->pAccelLabel);
    d_particleState[m].push_back(lb->pVelocityLabel);
    d_particleState[m].push_back(lb->pMassLabel);
    d_particleState[m].push_back(lb->pChargeLabel);
    d_particleState[m].push_back(lb->pParticleIDLabel);
  }
}

void MD::scheduleSPME()
{
  // Global "driver" routine:
  //  Variables prepended with f exist only in fourier space,
  //  variables prepended with r exist only in real space, otherwise they may be either.

  // Extract needed values from the system description
  IntVector SPMEGridExtents;
  Matrix3 InverseCell;
  double SystemVolume;

  SPMEGridExtents = MD_System->GetSPMEGridExtents();
  InverseCell = MD_System->GetUnitCellInverse();
  SystemVolume = MD_System->GetUnitCellVolume();

  // Extract needed values from the local subgrid
  IntVector LocalGridExtents, LocalGridOffset;

  LocalGridExtents = PatchLocalGrid->GetGridExtents();
  LocalGridOffset = PatchLocalGrid->GetGlobalOffset();

  SimpleGrid fStressPre, fTheta;
  // These should persist through mapping from global to local system; One global map should be kept and only change
  //   the box or grid points change.
  IntVector LocalGridExtents, LocalGridOffset;

  // Initialize some things we'll need repeatedly
  // Generate the local vectors of m_i
  vector<double> M1(LocalGridExtents.x()), M2(LocalGridExtents.y()), M3(LocalGridExtents.z());

  M1 = GenerateMVector(LocalGridExtents.x(), LocalGridOffset.x(), SPMEGridExtents.x());
  M2 = GenerateMVector(LocalGridExtents.y(), LocalGridOffset.y(), SPMEGridExtents.y());
  M3 = GenerateMVector(LocalGridExtents.z(), LocalGridOffset.z(), SPMEGridExtents.z());
  if (NewBox) {  // Box dimensions have changed, we need to update B and C
    SimpleGrid fBGrid, fCGrid;
    CalculateStaticGrids(LocalGridExtents, MD_System, fBGrid, fCGrid, fStressPre);
    fThetaRecip = fBGrid * fCGrid;  // Multiply per point
  }
  MapLocalGrid (LocalGridMap);

  bool converged = false;  // Calculate at least once
  while (!converged) {  // Iterate over this subset until convergence is reached
    Q.MapCharges(LocalGrid, LocalGridMap);
    Q.Transform(RealToFourier);

    Matrix3 StressTensor_Local;  // Accumulate local -> global
    double Energy_Local;  // Accumulate local -> global
    Q.CalculateEnergyAndStress(Energy_Local, StressTensorLocal, StressPre, ThetaRecip);  // Q^ = Q^*B*C by the end of this
    Q.Transform(FourierToReal);
    // Q.CalculatePolarization(CurrentPolarizationGrid,OldPolarizationGrid);
    // Check polarization convergence
    converged = true;
    if (polarizable) {
      converged = CheckConvergence(CurrentPolarizationGrid, OldPolarizationGrid);
    }
  }
  Q.ExtractForces(LocalGridMap, ParticleList);   // Map the forces back onto the particles;
}

void MD::extractCoordinates()
{
  std::ifstream inputFile;
  inputFile.open(coordinateFile_.c_str());
  if (!inputFile.is_open()) {
    string message = "\tCannot open input file: " + coordinateFile_;
    throw ProblemSetupException(message, __FILE__, __LINE__);
  }

  // do file IO to extract atom coordinates
  string line;
  unsigned int numRead;
  for (unsigned int i = 0; i < numAtoms_; i++) {
    // get the atom coordinates
    getline(inputFile, line);
    double x, y, z;
    numRead = sscanf(line.c_str(), "%lf %lf %lf", &x, &y, &z);
    if (numRead != 3) {
      string message = "\tMalformed input file. Should have [x,y,z] coordinates per line: ";
      throw ProblemSetupException(message, __FILE__, __LINE__);
    }
    Point pnt(x, y, z);
    atomList.push_back(pnt);
  }
  inputFile.close();
}

void MD::generateNeighborList()
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffRadius_ * cutoffRadius_;
  for (unsigned int i = 0; i < numAtoms_; i++) {
    for (unsigned int j = 0; j < numAtoms_; j++) {
      if (i != j) {
        // the vector distance between atom i and j
        reducedCoordinates = atomList[i] - atomList[j];

        // this is required for periodic boundary conditions
        reducedCoordinates -= (reducedCoordinates / box_).vec_rint() * box_;

        // eliminate atoms outside of cutoff radius, add those within as neighbors
        if ((fabs(reducedCoordinates[0]) < cutoffRadius_) && (fabs(reducedCoordinates[1]) < cutoffRadius_)
            && (fabs(reducedCoordinates[2]) < cutoffRadius_)) {
          double reducedX = reducedCoordinates[0] * reducedCoordinates[0];
          double reducedY = reducedCoordinates[1] * reducedCoordinates[1];
          double reducedZ = reducedCoordinates[2] * reducedCoordinates[2];
          r2 = sqrt(reducedX + reducedY + reducedZ);
          // only add neighbor atoms within spherical cut-off around atom "i"
          if (r2 < cut_sq) {
            neighborList[i].push_back(j);
          }
        }
      }
    }
  }
}

vector<Point> MD::calcReducedCoords(const vector<Point>& localRealCoordinates,
                                    const Transformation3D& Invert_Space)
{

  vector<Point> localReducedCoords;

  if (!Orthorhombic)  // bool Orthorhombic; true if simulation cell is orthorhombic, false if it's generic
    for (size_t Index = 0; Index < NumParticlesInCell; ++Index) {
      CoordType s;        // Fractional coordinates; 3 - vector
      s = ParticleList[Index].GetCoordinates();   // Get non-ghost particle coordinates for this cell
      s *= InverseBox;       // For generic coordinate systems; InverseBox is a 3x3 matrix so this is a matrix multiplication = slow
      Local_ReducedCoords.push_back(s);   // Reduced non-ghost particle coordinates for this cell
    }
  else
    for (size_t Index = 0; Index < NumParticlesInCell; ++Index) {
      CoordType s;        // Fractional coordinates; 3-vector
      s = ParticleList[Index].GetCoordinates();   // Get non-ghost particle coordinates for this cell
      s(0) *= Invert_Space(0, 0);
      s(1) *= Invert_Space(1, 1);
      s(2) *= Invert_Space(2, 2);                // 6 Less multiplications and additions than generic above
      Local_ReducedCoords.push_back(s);         // Reduced non-ghost particle coordinates for this cell
    }

  return Local_ReducedCoords;
}

bool MD::isNeighbor(const Point* atom1,
                    const Point* atom2)
{
  double r2;
  Vector reducedCoordinates;
  double cut_sq = cutoffRadius_ * cutoffRadius_;

  // the vector distance between atom 1 and 2
  reducedCoordinates = *atom1 - *atom2;

  // this is required for periodic boundary conditions
  reducedCoordinates -= (reducedCoordinates / box_).vec_rint() * box_;

  // check if outside of cutoff radius
  if ((fabs(reducedCoordinates[0]) < cutoffRadius_) && (fabs(reducedCoordinates[1]) < cutoffRadius_)
      && (fabs(reducedCoordinates[2]) < cutoffRadius_)) {
    r2 = sqrt(pow(reducedCoordinates[0], 2.0) + pow(reducedCoordinates[1], 2.0) + pow(reducedCoordinates[2], 2.0));
    return r2 < cut_sq;
  }
  return false;
}

void MD::initialize(const ProcessorGroup* /* pg */,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* /*old_dw*/,
                    DataWarehouse* new_dw)
{
  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // get bounds of current patch to correctly initialize particles (atoms)
    IntVector low = patch->getExtraCellLowIndex();
    IntVector high = patch->getExtraCellHighIndex();

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleVariable<Point> px;
      ParticleVariable<Vector> pforce;
      ParticleVariable<Vector> paccel;
      ParticleVariable<Vector> pvelocity;
      ParticleVariable<double> penergy;
      ParticleVariable<double> pmass;
      ParticleVariable<double> pcharge;
      ParticleVariable<long64> pids;

      // eventually we'll need to use PFS for this
      vector<Point> localAtoms;
      for (unsigned int i = 0; i < numAtoms_; i++) {
        if (containsAtom(low, high, atomList[i])) {
          localAtoms.push_back(atomList[i]);
        }
      }

      ParticleSubset* pset = new_dw->createParticleSubset(localAtoms.size(), matl, patch);
      new_dw->allocateAndPut(px, lb->pXLabel, pset);
      new_dw->allocateAndPut(pforce, lb->pForceLabel, pset);
      new_dw->allocateAndPut(paccel, lb->pAccelLabel, pset);
      new_dw->allocateAndPut(pvelocity, lb->pVelocityLabel, pset);
      new_dw->allocateAndPut(penergy, lb->pEnergyLabel, pset);
      new_dw->allocateAndPut(pmass, lb->pMassLabel, pset);
      new_dw->allocateAndPut(pcharge, lb->pChargeLabel, pset);
      new_dw->allocateAndPut(pids, lb->pParticleIDLabel, pset);

      int numParticles = pset->numParticles();
      for (int i = 0; i < numParticles; i++) {
        Point pos = localAtoms[i];
        px[i] = pos;
        pforce[i] = Vector(0.0, 0.0, 0.0);
        paccel[i] = Vector(0.0, 0.0, 0.0);
        pvelocity[i] = Vector(0.0, 0.0, 0.0);
        penergy[i] = 0.0;
        pmass[i] = 2.5;
        pcharge[i] = 0.0;
        pids[i] = patch->getID() * numAtoms_ + i;

        // TODO update this with new VarLabels
        if (md_dbg.active()) {
          std::cout.setf(std::ios_base::showpoint);  // print decimal and trailing zeros
          std::cout.setf(std::ios_base::left);  // pad after the value
          std::cout.setf(std::ios_base::uppercase);  // use upper-case scientific notation
          std::cout << std::setw(10) << "Patch_ID: " << std::setw(4) << patch->getID();
          std::cout << std::setw(14) << " Particle_ID: " << std::setw(4) << pids[i];
          std::cout << std::setw(12) << " Position: " << pos;
          std::cout << std::endl;
        }
      }
    }
    new_dw->put(sum_vartype(0.0), lb->vdwEnergyLabel);
  }
}

vector<double> MD::generateMVector(int Points,
                                   int Shift,
                                   int Max)
{
  vector<double> M(Points);
  int HalfMax = Max / 2;

  for (size_t m = 0; m < Points; ++m) {
    M[m] = m + Shift;
    if (M[m] > HalfMax)
      M[m] -= Max;
  }
  return M;
}

vector<std::complex<double> > MD::generateBVector(const int& points,
                                                  const vector<double>& M,
                                                  const int& max,
                                                  const int& splineOrder,
                                                  const vector<double>& splineCoeff)
{
  double PI = acos(-1.0);
  double OrderM12PI = (splineOrder - 1) * 2.0 * PI;

  vector<complex<double>> b(points);
  for (size_t Ind = 0; Ind < points; ++Ind) {
    double k = M[Ind] / max;
    complex<double> Numerator = complex(cos(OrderM12PI * k), sin(OrderM12PI * k));
    complex<double> Denominator;
    for (size_t p = 0; p < splineOrder - 1; ++p) {
      Denominator += splineCoeff[p] * complex(cos(OrderM12PI * k1 * p), sin(OrderM12PI * k1 * p));
    }
    b[Ind] = Numerator / Denominator;
  }
  return b;
}

void MD::calculateStaticGrids(const SubGrid& LocalGrid,
                              const System_Reference& MD_System,
                              SimpleGrid& fB,
                              SimpleGrid& fC,
                              SimpleGrid& StressPreMult)
{
  Matrix3 InverseUnitCell;
  double Ewald_Beta;
  IntVector GridExtents, SubGridOffset, K;

  IntVector HalfGrid = GridExtents / 2;

  InverseUnitCell = MD_System->CellInverse();
  EwaldBeta = MD_System->EwaldDampingCoefficient();

  GridExtents = LocalGrid->GetExtents();          // Extents of the local subgrid
  SubGridOffset = LocalGrid->GetOffsetVector();     // Offset needed to map local subgrid into global grid
  K = LocalGrid->GetGlobalExtent();     // Global grid size (K1,K2,K3)

  vector<double> M1, M2, M3;
  // Generate vectors of m_i/K_i

  M1 = GenerateMVector(GridExtents.x(), SubGridOffset.x(), K.x());
  M2 = GenerateMVector(GridExtents.y(), SubGridOffset.y(), K.y());
  M3 = GenerateMVector(GridExtents.z(), SubGridOffset.z(), K.z());

  vector<double> OrdinalSpline(SplineOrder - 1);
  OrdinalSpline = GenerateOrdinalSpline(SplineOrder);

  vector<complex<double>> b1, b2, b3;
  // Generate vectors of b_i (=exp(i*2*Pi*(n-1)m_i/K_i)*sum_(k=0..p-2)M_n(k+1)exp(2*Pi*k*m_i/K_i)

  b1 = GeneratebVector(GridExtents.x(), M1, K.x(), SplineOrder, OrdinalSpline);
  b2 = GeneratebVector(GridExtents.y(), M1, K.y(), SplineOrder, OrdinalSpline);
  b3 = GeneratebVector(GridExtents.z(), M1, K.z(), SplineOrder, OrdinalSpline);

  // Use previously calculated vectors to calculate our grids
  double PI, PI2, InvBeta2, VolFactor

  PI = acos(-1.0);
  PI2 = PI * PI;
  InvBeta2 = 1.0 / (Ewald_Beta * Ewald_Beta);
  VolFactor = 1.0 / (PI * MD_System->GetVolume());

  for (size_t kX = 0; kX < GridExtents.x(); ++kX) {
    for (size_t kY = 0; kY < GridExtents.y(); ++kY) {
      for (size_t kZ = 0; kZ < GridExtents.z(); ++kZ) {

        fB[kX][kY][kZ] = norm(b1[kX]) * norm(b2[kY]) * norm(b3[kZ]);  // Calculate B

        if (kX != kY != kZ != 0) {  // Calculate C and stress premultiplication factor
          Vector m(M1[kX], M2[kY], M3[kZ]), M;
          double M2, Factor;

          M = m * InverseUnitCell;
          M2 = M.length2();
          Factor = PI2 * M2 * InvBeta2;

          fC[kX][kY][kZ] = VolFactor * exp(-Factor) / M2;
          StressPreMult[kX][kY][kZ] = 2 * (1 + Factor) / M2;
        }
      }
    }
  }
  fC[0][0][0] = 0.0;
  StressPremult[0][0][0] = 0.0;  // Exceptional values
}

void MD::solveSPMECharge()
{
  // Outside of iterations, map particle positions to grid charge contributions -
  SPMEGridMap CurrentPatchGridMap = createSPMEChargeMap(CurrentSPMEGrid, Current_Patch, MD_System);

  // Iterative procedure would begin here
  CurrentPatchGridMap.MapAllCharges(CurrentSPMEGrid, Current_Patch);  // Maps the charges from particles to the SPME_Grid
  CurrentSPMEGrid.RealToFourier();
  Energy_Recip = CurrentSPMEGrid.CalculateEnergy(ThetaRecip);  // Q*F(B.C)

//  Data types
//    SPME_Grid:          Contains just the necessary information for the SPME routine;  Grid points, grid extent, charge per point, energy(?)
//    (Global)            Global, must rectify it through all processors before and after FFT routine and within polarizability iteration
//                        Also contains global system information about SPME (MeshPointLimits, SplineOrder, etc...)
//
//    SPME_Grid_Map:      Local data structure, one per patch.  Has a vector of map data per grid point, projected from the particles in the system.
//    (Local, PerPatch)   This constitutes a potential race condition (Adding to the same grid point data map from multiple particles in the same patch).
//                        Data structure extends over all the grid points belonging to the local patch, plus some extension of "ghost" points dealing with
//                        particles which map to points outside the spatial boundary of the current patch.
//                        Contains:  particle charge maps, offset (to register with global spatial grid)
//
//    ParticleIterator:   Some type of iterator through the particles local to the patch.
//    (Local, PerPatch)
//
//    SystemReference:    System data for MD simulations; contains things like the Cell matrix which codifies the size of the entire simulation cell, the
//    (Global)            inverse cell matrix.
}

SPMEGrid MD::SPME_Initialize(const IntVector& EwaldMeshLimits,
                             const Matrix3D& CellInverse,
                             const Matrix3D& Cell,
                             const double& EwaldScale,
                             const int& SplineOrder)
{
  IntVector K, HalfK;
  K = EwaldMeshLimits;
  HalfK = K / 2;

  Vector KInverse = 1.0 / K;

  double PiSquared = PI * PI;
  double InvBetaSquared = 1.0 / (EwaldScale * EwaldScale);

  SPME_Grid B, C;
  // Calculate the C array
  if (Orthorhombic) {
    double CellVolume, As1, As2, As3;
    CellVolume = Cell(0, 0) * Cell(1, 1) * Cell(2, 2);
    As1 = CellInverse(0, 0);
    As2 = CellInverse(1, 1);
    As3 = CellInverse(2, 2);

    double CPreFactor = 1.0 / (PI * CellVolume);

    for (int l1 = 0; l1 <= HalfK[0]; ++l1) {
      double m1 = l1 * As1;
      double m1OverK1 = m1 * KInverse[0];
      double MSquared = m1 * m1;
      for (int l2 = 0; l2 <= HalfK[1]; ++l2) {
        double m2 = l2 * As2;
        double m2OverK2 = m2 * KInverse[1];
        MSquared += m2 * m2;
        for (int l3 = 0; l3 <= HalfK[2]; ++l3) {
          double m3 = l3 * As3;
          double m3OverK3 = m3 * KInverse[3];
          MSquared += m3 * m3;

          double C = CPreFactor * exp(-PiSquared * MSquared * InvBetaSquared) / MSquared;
        }
      }
    }
  }

  Vector& A1, A2, A3;
  A1 = Cell.ExtractRow(0);  // Assumes the vectors in Cell are stored in rows
  A2 = Cell.ExtractRow(1);
  A3 = Cell.ExtractRow(2);

  Vector& As1, As2, As3;
  As1 = Cell.ExtractCol(0);  // Assumes the vectors in CellInverse are stored in columns
  As2 = Cell.ExtractCol(1);
  As3 = Cell.ExtractCol(2);

  double CellVolume = (A1.Cross(A2)).Dot(A3);
  // C(m1,m2,m3) = (1/PI*V)*(exp(-PI^2*M^2/Beta^2)/M^2)

  double CPreFactor = 1.0 / (PI * CellVolume);

  int K1 = EwaldMeshLimits[0];
  int HalfK1 = K1 / 2;
  int K2 = EwaldMeshLimits[1];
  int HalfK2 = K2 / 2;
  int K3 = EwaldMeshLimits[2];
  int HalfK3 = K3 / 2;

  SPME_Grid FieldGrid;
  double PiSquared = PI * PI;
  double OneOverBeta2 = 1.0 / (EwaldScale * EwaldScale);

  // Orthorhombic
  if (Orthorhombic) {
    for (int m1 = 0; m1 <= HalfK1; ++m1) {
      for (int m2 = 0; m2 <= HalfK2; ++m2) {
        for (int m3 = 0; m3 <= HalfK3; ++m3) {
          double MSquared = m1 * As1 * m1 * As2 + m2 * As2 * m2 * As2 + m3 * As3 * m3 * As3;
          FieldGrid[m1][m2][m3] = CPreFactor * exp(-PiSquared * MSquared * OneOverBeta2) / MSquared;
        }
      }
    }
  }

}

template<class T>
SPMEGridMap MD::createSPMEChargeMap<T>(const SPMEGrid& SPMEGlobalGrid,
                                       const Patch& CurrentPatch,
                                       const SystemReference& SystemData)
{
  // Note:  SubGridOffset maps the offset of the current patch's subgrid to the global grid numbering scheme.
  //        For example, a patch that iterated from global grid point 3,4,5 to 7,8,9 would have a SubGridOffset
  //        of:  {3,4,5}.

  IntVector EwaldMeshLimits = SPME_GlobalGrid->GetMeshLimits();
  IntVector SplineOrder = SPME_GlobalGrid->GetSplineOrder();

  Vector PatchOffset = CurrentPatch->SpatialOffset();
  Vector PatchExtent = CurrentPatch->SpatialExtent();  // HighCorner-LowCorner, where HighCorner is the max(X,Y,Z) of the subspace, LowCorner is
                                                       //   min(X,Y,Z)

  ParticleIterator ParticleList = CurrentPatch->ParticleVector();

  Matrix3 CellInverse = SystemData->CellInverse();

  Vector InverseMeshLimits = 1.0 / EwaldMeshLimits;

  IntVector SubGridIndexOffset, SubGridIndexExtent;
  {
    // Contain these temp variables
    Vector TempOffset = PatchOffset * CellInverse;
    Vector TempExtent = PatchExtent * CellInverse;

    for (size_t Ind = 0; Ind < 3; ++Ind) {
      SubGridIndexOffset = floor(TempOffset[Ind] * static_cast<double>(EwaldMeshLimits[Ind]));  // Generate index offset to map local grid to global
      SubGridIndexExtent = floor(TempExtent[Ind] * static_cast<double>(EwaldMeshLimits[Ind]));  // Generate index count to determine # grid points inside patch
                                                                                                //   not including ghost grid points
    }
  }
  SPME_Grid_Map LocalGrid[SubGridIndexExtent[0] + SplineOrder][SubGridIndexExtent[1] + SplineOrder][SubGridIndexExtent[2]
                                                                                                    + SplineOrder];
  // 3D array of Vector<SPME_Map> type, messy data structure.  Suggestions?
  // Goal here is to create a sub-grid which maps to the patch local + extended "ghost" grid points, and to save the particle to grid charge mapping coefficients
  // so that we don't have to re-generate them again.  This step is essentially necessary every time step, or every time the global "Solve SPME Charge" routine
  // is done.

  int Global_Shift = 0;
  if (SystemData->Orthorhombic()) {  //  Take a few calculation shortcuts to save some matrix multiplication.  Worth it?  Not sure..
    Vector InverseGridMapping = EwaldMeshLimits * CellInverse;  // Orthorhomibc, so CellInverse is diagonal and we can pre-multiply
    for (ParticlePointer = ParticleList.begin(); ParticlePointer != ParticleList.end(); ++ParticlePointer) {
      Vector U_Current = ParticlePointer->GetCoordinateVector();                          // Extract coordinate vector from particle
      U_Current *= InverseGridMapping;                                                        // Convert to reduced coordinates
      IntVector CellIndex;
      for (size_t Ind = 0; Ind < 3; ++Ind) {
        CellIndex[Ind] = floor(U_Current[Ind]) - SubGridIndexOffset[Ind] + Global_Shift;
      }  // Vector floor + shift by element

      vector<Vector> Coeff_Array(SplineOrder), Deriv_Array(SplineOrder);
      CalculateSpline(U_Current, Coeff_Array, Deriv_Array, SplineOrder);
      for (int IndX = 0; IndX < SplineOrder; ++IndX) {
        for (int IndY = 0; IndY < SplineOrder; ++IndY) {
          for (int IndZ = 0; IndZ < SplineOrder; ++IndZ) {
            double Coefficient = Coeff_Array[IndX].x * Coeff_Array[IndY].y * Coeff_Array[IndZ].z;  // Calculate appropriate coefficient
            Vector Gradient(Deriv_Array[IndX].x, Deriv_Array[IndY].y, Deriv_Array[IndZ].z);  // Extract appropriate derivative vector
            (LocalGrid[CellIndex.x + IndX][CellIndex.y + IndY][CellIndex.z + IndZ]).AddMapPoint(ParticlePointer->GetGlobalHandle(),
                                                                                                Coefficient, Gradient);
          }
        }
      }
    }
  } else {
    for (ParticlePointer = ParticleList.begin(); ParticlePointer != ParticleList.end(); ++ParticlePointer) {
      Vector U_Current = ParticlePointer->GetCOordinateVector();  // Extract coordinate vector from particle
      U_Current *= CellInverse;                                    // Full matrix multiplication to get (X,Y,Z) for non-orthorhombic
      for (size_t Ind = 0; Ind < 3; ++Ind) {
        U_Current[Ind] *= EwaldMeshLimits[Ind];
      }
      for (size_t Ind = 0; Ind < 3; ++Ind) {
        CellIndex[Ind] = floor(U_Current[Ind]) + Global_Shift;
      }    // Vector floor + shift by element

      vector<Vector> Coeff_Array(SplineOrder), Deriv_Array(SplineOrder);
      CalculateSpline(U_Current, Coeff_Array, Deriv_Array, SplineOrder);
      for (int IndX = 0; IndX < SplineOrder; ++IndX) {
        for (int IndY = 0; IndY < SplineOrder; ++IndY) {
          for (int IndZ = 0; IndZ < SplineOrder; ++IndZ) {
            double Coefficient = Coeff_Array[IndX].x * Coeff_Array[IndY].y * Coeff_Array[IndZ].z;  // Calculate appropriate coefficient
            Vector Gradient(Deriv_Array[IndX].x, Deriv_Array[IndY].y, Deriv_Array[IndZ].z);  // Extract appropriate derivative vector
            (LocalGrid[CellIndex.x + IndX][CellIndex.y + IndY][CellIndex.z + IndZ]).AddMapPoint(ParticlePointer->GetGlobalHandle(),
                                                                                                Coefficient, Gradient);
          }
        }
      }
    }
  }
  return null;
}

void MD::spmeMapChargeToGrid(SPMEGrid<std::complex<double> >& LocalGridCopy,
                             const SPMEGridMap& LocalGridMap,
                             const Patch& CurrentPatch)
{
  IntVector Extent = LocalGridMap.GetLocalExtents();  // Total number of grid points on the local grid including ghost points (X,Y,Z)
  IntVector Initial = Extent + LocalGridMap.GetLocalShift();  // Offset of lowest index points in LOCAL coordinate system.
                                                              //  e.g. for SplineOrder = N, indexing from -N/2 to X + N/2 would have a shift of N/2 and an extent
                                                              //       of X+N, indexing from 0 to N would have a shift of 0 and an extent of X+N

  ParticleIterator ParticleList = CurrentPatch->ParticleVector();

  for (size_t X = Initial[0]; X < Extent[0]; ++X) {
    for (size_t Y = Initial[1]; Y < Extent[1]; ++Y) {
      for (size_t Z = Initial[2]; Z < Extent[2]; ++Z) {
        (LocalGridCopy[X][Y][Z]).IncrementGridCharge((LocalGridMap[X][Y][Z]).MapChargeFromAtoms(ParticleList));
      }
    }
  }
}

SPMEGrid MD::fC(const IntVector& GridExtents,
                const System_Reference& MD_System)
{
  Matrix3 InverseCell;
  SimpleGrid<double> C(GridExtents.x, GridExtents.y, GridExtents.z);
  double Ewald_Beta;

  InverseCell = MD_System->CellInverse();
  Ewald_Beta = MD_System->GetEwaldBeta();

  IntVector HalfGrid = GridExtents / 2;

  double PI = acos(-1.0);
  double PI2 = PI * PI;
  double InvBeta2 = 1.0 / (Ewald_Beta * Ewald_Beta);

  double VolFactor = 1.0 / (PI * MD_System->GetVolume());

  Vector M;
  for (size_t m1 = 0; m1 < GridExtents.x(); ++m1) {
    M[0] = m1;
    if (m1 > HalfGrid.x())
      M[0] -= GridExtents.x();
    for (size_t m2 = 0; m2 < GridExtents.y(); ++m2) {
      M[1] = m2;
      if (m2 > HalfGrid.y())
        M[1] -= GridExtents.y();
      for (size_t m3 = 0; m3 < GridExtents.z(); ++m3) {
        M[2] = m3;
        if (m3 > HalfGrid.z())
          M[2] -= GridExtents.z();
        // Calculate C point values
        if (!(m1 == 0) && !(m2 == 0) && !(m3 == 0)) {  // Discount C(0,0,0)
          double TempM = M * InverseCell;
          double M2 = TempM.length2(TempM);
          double Val = VolFactor * exp(-PI2 * M2 * InvBeta2) / M2;
          C[m1][m2][m3] = Val;
        }
      }
    }
  }
  C[m1][m2][m3] = 0.0;
  return C;
}

SPMEGrid MD::fB(const IntVector& GridExtents,
                const System_Reference& MD_System)
{
  Matrix3 InverseCell;
  SimpleGrid B(GridExtents.x(), GridExtents.y(), GridExtents, z());

  InverseCell = MD_System->CellInverse();
  IntVector HalfGrid = GridExtents / 2;
  Vector InverseGrid = 1.0 / GridExtents;

  double PI = acos(-1.0);
  double OrderM12PI = 2.0 * PI * (SplineOrder - 1);

  vector<complex<double> > b1(GridExtents.x()), b2(GridExtents.y()), b3(GridExtents.z());
  vector<complex<double> > OrdinalSpline(SplineOrder - 1);

  OrdinalSpline = CalculateOrdinalSpline(SplineOrder - 1, SplineOrder);  // Calculates Mn(0)..Mn(n-1)

  // Calculate k_i = m_i/K_i
  for (size_t m1 = 0; m1 < GridExtents.x(); ++m1) {
    double kX = m1;
    if (m1 > HalfGrid.x())
      kX = m1 - GridExtents.x();
    kX /= GridExtents.x();
    complex<double> num = complex(cos(OrderM12PI * kX), sin(OrderM12PI * kX));
    complex<double> denom = OrdinalSpline[0];  //
    for (size_t k = 1; k < SplineOrder - 1; ++k) {
      denom += OrdinalSpline[k] * complex(cos(OrderM12PI * kX * k), sin(OrderM12PI * kX * k));
    }
    b1[m1] = num / denom;
  }

  for (size_t m2 = 0; m2 < GridExtents.y(); ++m2) {
    double kY = m2;
    if (m2 > HalfGrid.y())
      kY = m2 - GridExtents.y();
    kY /= GridExtents.y();
    complex<double> num = complex(cos(OrderM12PI * kY), sin(OrderM12PI * kY));
    complex<double> denom = OrdinalSpline[0];  //
    for (size_t k = 1; k < SplineOrder - 1; ++k) {
      denom += OrdinalSpline[k] * complex(cos(OrderM12PI * kY * k), sin(OrderM12PI * kY * k));
    }
    b2[m2] = num / denom;
  }

  for (size_t m3 = 0; m3 < GridExtents.z(); ++m3) {
    double kZ = m3;
    if (m3 > HalfGrid.z())
      kZ = m3 - GridExtents.y();
    kZ /= GridExtents.y();
    complex<double> num = complex(cos(OrderM12PI * kZ), sin(OrderM12PI * kZ));
    complex<double> denom = OrdinalSpline[0];  //
    for (size_t k = 1; k < SplineOrder - 1; ++k) {
      denom += OrdinalSpline[k] * complex(cos(OrderM12PI * kZ * k), sin(OrderM12PI * kZ * k));
    }
    b3[m3] = num / denom;
  }

  for (size_t m1 = 0; m1 < GridExtents.x(); ++m1) {
    for (size_t m2 = 0; m2 < GridExtents.x(); ++m2) {
      for (size_t m3 = 0; m3 < GridExtents.x(); ++m3) {
        // Calculate B point values
        B[m1][m2][m3] = norm(b1[m1]) * norm(b2[m2]) * norm(b3[m3]);
      }
    }
  }

}

void MD::calculateNonBondedForces(const ProcessorGroup* pg,
                                  const PatchSubset* patches,
                                  const MaterialSubset* matls,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw)
{
  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    double vdwEnergy = 0;
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleSubset* pset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // requires variables
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pforce;
      constParticleVariable<double> penergy;
      constParticleVariable<long64> pids;
      old_dw->get(px, lb->pXLabel, pset);
      old_dw->get(penergy, lb->pEnergyLabel, pset);
      old_dw->get(pforce, lb->pForceLabel, pset);
      old_dw->get(pids, lb->pParticleIDLabel, pset);

      // computes variables
      ParticleVariable<Vector> pforcenew;
      ParticleVariable<double> penergynew;
      new_dw->allocateAndPut(penergynew, lb->pEnergyLabel_preReloc, pset);
      new_dw->allocateAndPut(pforcenew, lb->pForceLabel_preReloc, pset);

      unsigned int numParticles = pset->numParticles();
      for (unsigned int i = 0; i < numParticles; i++) {
        pforcenew[i] = pforce[i];
        penergynew[i] = penergy[i];
      }

      // loop over all atoms in system, calculate the forces
      double r2, ir2, ir6, ir12, T6, T12;
      double forceTerm;
      Vector totalForce, atomForce;
      Vector reducedCoordinates;
      unsigned int totalAtoms = pset->numParticles();
      for (unsigned int i = 0; i < totalAtoms; i++) {
        atomForce = Vector(0.0, 0.0, 0.0);

        // loop over the neighbors of atom "i"
        unsigned int idx;
        unsigned int numNeighbors = neighborList[i].size();
        for (unsigned int j = 0; j < numNeighbors; j++) {
          idx = neighborList[i][j];

          // the vector distance between atom i and j
          reducedCoordinates = px[i] - px[idx];

          // this is required for periodic boundary conditions
          reducedCoordinates -= (reducedCoordinates / box_).vec_rint() * box_;
          double reducedX = reducedCoordinates[0] * reducedCoordinates[0];
          double reducedY = reducedCoordinates[1] * reducedCoordinates[1];
          double reducedZ = reducedCoordinates[2] * reducedCoordinates[2];
          r2 = reducedX + reducedY + reducedZ;
          ir2 = 1.0 / r2;  // 1/r^2
          ir6 = ir2 * ir2 * ir2;  // 1/r^6
          ir12 = ir6 * ir6;  // 1/r^12
          T12 = R12_ * ir12;
          T6 = R6_ * ir6;
          penergynew[idx] = T12 - T6;  // energy
          vdwEnergy += penergynew[idx];  // count the energy
          forceTerm = (12.0 * T12 - 6.0 * T6) * ir2;  // the force term
          totalForce = forceTerm * reducedCoordinates;

          // the contribution of force on atom i
          atomForce += totalForce;
        }  // end neighbor loop for atom "i"

        // sum up contributions to force for atom i
        pforcenew[i] += atomForce;

        if (md_dbg.active()) {
          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
          std::cout << "ParticleID: " << std::setw(6) << pids[i] << std::setw(6);
          std::cout << "Prev Position: [";
          std::cout << std::setw(10) << std::setprecision(4) << px[i].x();
          std::cout << std::setw(10) << std::setprecision(4) << px[i].y();
          std::cout << std::setprecision(10) << px[i].z() << std::setw(4) << "]";
          std::cout << "Energy: ";
          std::cout << std::setw(14) << std::setprecision(6) << penergynew[i];
          std::cout << "Force: [";
          std::cout << std::setw(14) << std::setprecision(6) << pforcenew[i].x();
          std::cout << std::setw(14) << std::setprecision(6) << pforcenew[i].y();
          std::cout << std::setprecision(6) << pforcenew[i].z() << std::setw(4) << "]";
          std::cout << std::endl;
        }
      }  // end atom loop

      // this accounts for double energy with Aij and Aji
      vdwEnergy *= 0.50;

      if (md_dbg.active()) {
        Vector forces(0.0, 0.0, 0.0);
        for (unsigned int i = 0; i < numParticles; i++) {
          forces += pforcenew[i];
        }
        std::cout.setf(std::ios_base::scientific);
        std::cout << "Total Local Energy: " << std::setprecision(16) << vdwEnergy << std::endl;
        std::cout << "Local Force: [";
        std::cout << std::setw(16) << std::setprecision(8) << forces.x();
        std::cout << std::setw(16) << std::setprecision(8) << forces.y();
        std::cout << std::setprecision(8) << forces.z() << std::setw(4) << "]";
        std::cout << std::endl;
        std::cout.unsetf(std::ios_base::scientific);
      }

      new_dw->deleteParticles(delset);

    }  // end materials loop

    // global reduction on
    new_dw->put(sum_vartype(vdwEnergy), lb->vdwEnergyLabel);

  }  // end patch loop

}

void MD::updatePosition(const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw)
{
  // loop through all patches
  unsigned int numPatches = patches->size();
  for (unsigned int p = 0; p < numPatches; p++) {
    const Patch* patch = patches->get(p);

    // do this for each material; for this example, there is only a single material, material "0"
    unsigned int numMatls = matls->size();
    for (unsigned int m = 0; m < numMatls; m++) {
      int matl = matls->get(m);

      ParticleSubset* lpset = old_dw->getParticleSubset(matl, patch);
      ParticleSubset* delset = scinew ParticleSubset(0, matl, patch);

      // requires variables
      constParticleVariable<Point> px;
      constParticleVariable<Vector> pforce;
      constParticleVariable<Vector> paccel;
      constParticleVariable<Vector> pvelocity;
      constParticleVariable<double> pmass;
      constParticleVariable<double> pcharge;
      constParticleVariable<long64> pids;
      old_dw->get(px, lb->pXLabel, lpset);
      old_dw->get(pforce, lb->pForceLabel, lpset);
      old_dw->get(paccel, lb->pAccelLabel, lpset);
      old_dw->get(pvelocity, lb->pVelocityLabel, lpset);
      old_dw->get(pmass, lb->pMassLabel, lpset);
      old_dw->get(pcharge, lb->pChargeLabel, lpset);
      old_dw->get(pids, lb->pParticleIDLabel, lpset);

      // computes variables
      ParticleVariable<Point> pxnew;
      ParticleVariable<Vector> paccelnew;
      ParticleVariable<Vector> pvelocitynew;
      ParticleVariable<double> pmassnew;
      ParticleVariable<double> pchargenew;
      ParticleVariable<long64> pidsnew;
      new_dw->allocateAndPut(pxnew, lb->pXLabel_preReloc, lpset);
      new_dw->allocateAndPut(paccelnew, lb->pAccelLabel_preReloc, lpset);
      new_dw->allocateAndPut(pvelocitynew, lb->pVelocityLabel_preReloc, lpset);
      new_dw->allocateAndPut(pmassnew, lb->pMassLabel_preReloc, lpset);
      new_dw->allocateAndPut(pchargenew, lb->pChargeLabel_preReloc, lpset);
      new_dw->allocateAndPut(pidsnew, lb->pParticleIDLabel_preReloc, lpset);

      // get delT
      delt_vartype delT;
      old_dw->get(delT, d_sharedState_->get_delt_label(), getLevel(patches));

      // loop over the local atoms
      unsigned int localNumParticles = lpset->numParticles();
      for (unsigned int i = 0; i < localNumParticles; i++) {

        // carry these values over for now
        pmassnew[i] = pmass[i];
        pchargenew[i] = pcharge[i];
        pidsnew[i] = pids[i];

        // update position
        paccelnew[i] = pforce[i] / pmass[i];
        pvelocitynew[i] = pvelocity[i] + paccel[i] * delT;
        pxnew[i] = px[i] + pvelocity[i] + pvelocitynew[i] * 0.5 * delT;

        if (md_dbg.active()) {
          std::cout << "PatchID: " << std::setw(4) << patch->getID() << std::setw(6);
          std::cout << "ParticleID: " << std::setw(6) << pidsnew[i] << std::setw(6);
          std::cout << "New Position: [";
          std::cout << std::setw(10) << std::setprecision(6) << pxnew[i].x();
          std::cout << std::setw(10) << std::setprecision(6) << pxnew[i].y();
          std::cout << std::setprecision(6) << pxnew[i].z() << std::setw(4) << "]";
          std::cout << std::endl;
        }
      }  // end atom loop

      new_dw->deleteParticles(delset);

    }  // end materials loop

  }  // end patch loop
}
