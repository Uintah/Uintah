#pragma once
#include "Utils.h"
#include "BoundingBox.h"
#include "Solver.h"
#include "Particle.h"

void const_vel_gen(ParticleDataList& l, const BoundingBox& b);
void deformation_gen(ParticleDataList& l, const BoundingBox& b);
