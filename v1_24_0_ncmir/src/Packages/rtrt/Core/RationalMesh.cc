#include <Packages/rtrt/Core/RationalMesh.h>

using namespace rtrt;

RationalMesh::RationalMesh (int m, int n)
{
    msize = m;
    nsize = n;
    mesh = new Point4D *[m];

    init_comb_table();


    for (int i=0; i<m; i++)
    {
        mesh[i] = new Point4D[n];
    }
}

RationalMesh::~RationalMesh()
{
    int i;
    
    for (i=0; i<msize; i++) {
        delete mesh[i];
    }
    delete mesh;
}

RationalMesh * RationalMesh::Copy() {
    
    RationalMesh *m = new RationalMesh(msize,nsize);
    
    for (int i=0; i<msize; i++) 
        for (int j=0; j<nsize; j++) {
            m->mesh[i][j] = mesh[i][j];
        }
    return m;
}


