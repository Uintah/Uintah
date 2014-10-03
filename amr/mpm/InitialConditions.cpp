#include "InitialConditions.h"

void const_vel_gen(ParticleDataList& l, const BoundingBox& b)
{
    //generates N^2 equally spaced particles in the
    //lower left quadrant of the given bounding box
    Vec2D domain_from = b.GetFrom();
    Vec2D domain_to = b.GetTo();
    Vec2D p_to = domain_from + (domain_to - domain_from)/8.0;
    Vec2D da = p_to - domain_from;
    double area = da[x1]*da[x2];
    int N = 4;
    //for equally spaced: da/(N+1)
    Vec2D d = da/double(N+1);
    unsigned int id = 0;
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
        {
            Vec2D cur_pos = {domain_from[x1] + double(j+1)*d[x1], domain_from[x2] + double(i+1)*d[x2]};
            ParticleData data;//default constructor all fields are zero
            data.id = id;
            data.pos = cur_pos;
            data.vel = {1.0, 1.0};
            data.m = 1.0;
            data.V = area/(N*N);
            data.F = {1.0, 0.0, 0.0, 1.0};
            data.y_mod = 0.0001;
            l.push_back(data);
            id++;
        }
}

void deformation_gen(ParticleDataList& l, const BoundingBox& b)
{
    //a beam that occupies third of the domain vertically
    //and half of the domain horizontally
    Vec2D domain_from = b.GetFrom();
    Vec2D domain_to = b.GetTo();
    Vec2D p_to = {domain_from[x1] + (domain_to[x1] - domain_from[x1])/2.0, domain_from[x2] + (domain_to[x2] - domain_from[x2])/3.0*2.0};
    Vec2D p_from = {domain_from[x1], domain_from[x2] + (domain_to[x2] - domain_from[x2])/3.0};
    Vec2D da = p_to - p_from;
    double area = da[x1]*da[x2];
    int Nx = 10;
    int Ny = 8;
    unsigned int id = 0;
    double dx = da[x1]/double(Nx+1);
    double dy = da[x2]/double(Ny+1);
    for(int i = 0; i < Nx; i++)
        for(int j = 0; j < Ny; j++)
        {
            Vec2D cur_pos = {p_from[x1] + double(i+1)*dx, p_from[x2] + double(j+1)*dy};
            ParticleData data;//default constructor all fields are zero
            data.id = id;
            data.pos = cur_pos;
            data.m = 1.0;
            data.V = area/(Nx*Ny);
            data.F = {1.0, 0.0, 0.0, 1.0};
            data.y_mod = 0.0001;
            //external force if applied to the last column of particles
            if(i == Nx - 1 && j > Ny - 4)
            {
                data.b = {-0.5, 0.0};
            }
            l.push_back(data);
            id++;
        }
 }
