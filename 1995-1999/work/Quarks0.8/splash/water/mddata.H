typedef double vm_type[3];

typedef struct mol_dummy {
    vm_type VM;
    double F[MXOD2][NDIR][NATOM];
} molecule_type;

#define MAXPROCS 128
