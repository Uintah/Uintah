#pragma once

#include<vector>
#include<list>
#include<map>
#include<array>
#include<functional>

using namespace std;

//vectors
typedef array<double,2> Vec2D;
typedef array<double,4> Mat2D;

//element/node/particle data structures
struct Node;
struct Element;
struct ParticleData;
struct Particle;
class Basis;
class BoundingBox;

typedef vector<Element*> ElementPtrVector;
typedef list<Element*> ElementPtrList;
typedef list<Particle*> ParticlePtrList;
typedef map<int, ParticlePtrList> ElementIDToParticlePtrMap;
typedef list<Node> NodeList;
typedef list<Node*> NodePtrList;
typedef list<Vec2D> CoordList;
typedef array<Node*,2> InterpolationData;
typedef array<Basis*,4> LocalBasis;
typedef list<ParticleData> ParticleDataList;
//particle generator would be a function taking (and returning) the list 
//of particle data as well as the bounding box of the domain
typedef function<void(ParticleDataList&, const BoundingBox&)> ParticleGenerator;
