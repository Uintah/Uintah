/*
 *  FEM.cc:
 *
 *  Written by:
 *   veselin
 *   TODAY'S DATE HERE
 *
 */

#include "fem/fem.hpp"

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Geometry/Point.h>

#include <Dataflow/share/share.h>

namespace DDDAS
{

using namespace SCIRun;

class PSECORESHARE FEM : public Module
{
   private:
      GuiInt gui_method;
      GuiInt gui_poly_degree;
      GuiInt gui_iter_method;
      GuiInt gui_max_iter;
      GuiInt gui_restart_iter;
      GuiInt gui_print_iter;
      GuiDouble gui_nu;
      GuiDouble gui_rtol;
      FieldHandle outfieldhandle;

   public:
      FEM(GuiContext*);

      virtual ~FEM();

      virtual void execute();

      virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(FEM)
FEM::FEM(GuiContext* ctx)
  : Module("FEM", ctx, Source, "PDESolver", "DDDAS"),
    gui_method(ctx->subVar("method")),
    gui_poly_degree(ctx->subVar("poly_degree")),
    gui_iter_method(ctx->subVar("iter_method")),
    gui_max_iter(ctx->subVar("max_iter")),
    gui_restart_iter(ctx->subVar("restart_iter")),
    gui_print_iter(ctx->subVar("print_iter")),
    gui_nu(ctx->subVar("nu")),
    gui_rtol(ctx->subVar("rtol"))
{
   // empty
}

FEM::~FEM()
{
   // empty
}

void FEM::execute()
{
   FieldIPort* fin = (FieldIPort *)get_iport("Field Input");
   FieldHandle fld_handle;

   fin->get(fld_handle);

   if(!fld_handle.get_rep())
   {
      warning("No Data in 'Field Input' port.");
      return;
   }

   // must be a TetVolField<double>
   TetVolField<double> *tvd = 
      dynamic_cast<TetVolField<double> *>(fld_handle.get_rep());

   if (!tvd)
   {
      error("'Field Input' is not a TetVolField !");
      return;
   }
   TetVolMesh *m = tvd->get_typed_mesh().get_rep();
   TetVolMesh::Node::size_type nv; m->size(nv);
   TetVolMesh::Cell::size_type ne; m->size(ne);

   DMesh<3> *mesh = new DMesh<3> (nv, ne);

   TetVolMesh::Node::iterator niter; m->begin(niter);
   TetVolMesh::Node::iterator nend; m->end(nend);
   while (niter != nend)
   {
      double x[3];
      TetVolMesh::Node::index_type nidx = *niter;
      ++niter;

      SCIRun::Point p;
      m->get_center(p, nidx);
      x[0] = p.x(); x[1] = p.y(); x[2] = p.z();

      mesh->AddVertex (x);
   }

   TetVolMesh::Cell::iterator citer; m->begin(citer);
   TetVolMesh::Cell::iterator cend; m->end(cend);
   while (citer != cend)
   {
      int v[4];
      TetVolMesh::Cell::index_type nidx = *citer;
      ++citer;

      TetVolMesh::Node::array_type elem_vert;
      m->get_nodes (elem_vert, nidx);
      v[0] = elem_vert[0]; v[1] = elem_vert[1];
      v[2] = elem_vert[2]; v[3] = elem_vert[3];

//      TetVolMesh::Node::array_type::value_type * arr = &elem_vert[0];
      mesh->AddTet (v);
   }

   mesh->FinalizeTetMesh (1, 0);

   int  method        = 2;    // 1 - Standard FEM
                              // 2 - Baumann-Oden
                              // 3 - NIPG (Nu)
                              // 4 - IP (Nu)
   int  poly_degree   = 1;
   double Nu          = 1.0;

   gui_method.reset();
   method = gui_method.get();
   cout << "Using method " << method << endl;
   gui_nu.reset();
   Nu = gui_nu.get();
   cout << "Using Nu = " << Nu << endl;
   gui_poly_degree.reset();
   poly_degree = gui_poly_degree.get();
   cout << "Using polynomials of degree " << poly_degree << endl;

   FiniteElementCollection *fec;
   FiniteElementSpace *fespace;

   switch (method)
   {
      case 1:  // Std FEM
         switch (poly_degree)
         {
            case 1: fec = new LinearFECollection; break;
            case 2: fec = new QuadraticFECollection; break;
         }
         break;
      case 2:  // Baumann-Oden
      case 3:  // NIPG
      case 4:  // IP
         switch (poly_degree)
         {
            case 1: fec = new P1TetNonConfFECollection; break;
            case 2: fec = new P2Discont3DFECollection; break;
         }
         break;
   }

//   fec = new P1TetNonConfFECollection;

   fespace = new GeneralFiniteElementSpace(mesh, fec);

   cout << "Number of elements : " << mesh->GetNE() << '\n'
        << "Number of unknowns : " << fespace->GetNDofs() << endl;

   BilinearForm *a = new BilinearForm (fespace);
   LinearForm *b = new LinearForm (fespace);

   ConstantCoefficient PermCoeff (1.0);
   ConstantCoefficient DirCoeff (0.0);
   ConstantCoefficient SrcCoeff (1.0);
   switch (method)
   {
      case 1:  // Standard FEM
         a->AddDomainIntegrator (new LaplaceIntegrator (PermCoeff));
         break;

      case 2:  //  Baumann-Oden
         a->AddDomainIntegrator (new LaplaceIntegrator (PermCoeff));
         a->AddInteriorFaceIntegrator (new DGBOFaceIntegrator (&PermCoeff));
         //  Dirichlet b.c. for Baumann-Oden
         a->AddBdrFaceIntegrator (new DGBOFaceIntegrator (&PermCoeff));
         b->AddBdrFaceIntegrator (new DG1BoundaryIntegrator (&PermCoeff,
                                                             &DirCoeff));
         break;

      case 3:  //  NIPG (Nu)
         a->AddDomainIntegrator (new LaplaceIntegrator (PermCoeff));
         a->AddInteriorFaceIntegrator (new DGBOFaceIntegrator (&PermCoeff));
         a->AddInteriorFaceIntegrator (new DG0FaceIntegrator (Nu));
         //  Dirichlet b.c. for NIPG
         a->AddBdrFaceIntegrator (new DGBOFaceIntegrator (&PermCoeff));
         a->AddBdrFaceIntegrator (new DG0FaceIntegrator (Nu));
         b->AddBdrFaceIntegrator (new DG1BoundaryIntegrator (&PermCoeff,
                                                             &DirCoeff));
         b->AddBdrFaceIntegrator (new DG0BoundaryIntegrator (&DirCoeff, Nu));
         break;

      case 4:  // IP (Nu)
         a->AddDomainIntegrator (new LaplaceIntegrator (PermCoeff));
         a->AddInteriorFaceIntegrator (new DGIPFaceIntegrator (&PermCoeff));
         a->AddInteriorFaceIntegrator (new DG0FaceIntegrator (Nu));
         //  Dirichlet b.c. for IP
         a->AddBdrFaceIntegrator (new DGIPFaceIntegrator (&PermCoeff));
         a->AddBdrFaceIntegrator (new DG0FaceIntegrator (Nu));
         b->AddBdrFaceIntegrator (new DG1BoundaryIntegrator (&PermCoeff,
                                                             &DirCoeff, -1.0));
         b->AddBdrFaceIntegrator (new DG0BoundaryIntegrator (&DirCoeff, Nu));
         break;
   }

   //  Source
   b->AddDomainIntegrator(new DomainLFIntegrator(SrcCoeff));

   GridFunction *sol = new GridFunction(fespace);
   *sol = 0.0;

   cout << "Assembling the matrix and the rhs ... " << endl;
   cout << "A ... " << flush;
   a->Assemble();
   cout << "b ... " << flush;
   b->Assemble();
   if (method == 1)  // Standard FEM
   {
      cout << "b.c. ... " << flush;
      Array<int> attr_is_dir(1);
      attr_is_dir[0] = 1;
      Coefficient *c[1];
      c[0] = &DirCoeff;
      sol->ProjectBdrCoefficient (c, attr_is_dir);
      a->EliminateEssentialBC (attr_is_dir, *sol, *b);
   }
   cout << "finalize A ... " << flush;
   a->Finalize();
   cout << "done assembling." << endl;
   cout << "Non-zero entries in the matrix: "
        << a->SpMat().NumNonZeroElems() << endl
        << "Average number of entries per row: "
        << double(a->SpMat().NumNonZeroElems())/a->Size() << endl;

   Operator *prec;
   MatrixInverse *inv;

   int iter_method             = 2;
   int printit                 = 1;
   int maxit                   = 1000;
   int itermethod_rstart       = 50;
   double rtol                 = 1e-8;
   double atol                 = 0.0;

   gui_iter_method.reset();
   iter_method = gui_iter_method.get();
   cout << "Using iter. method " << iter_method << endl;
   gui_max_iter.reset();
   maxit = gui_max_iter.get();
   cout << "Max. iter. " << maxit << endl;
   gui_restart_iter.reset();
   itermethod_rstart = gui_restart_iter.get();
   gui_print_iter.reset();
   printit = gui_print_iter.get();
   gui_rtol.reset();
   rtol = gui_rtol.get();

//   prec = new GSSmoother(a->SpMat());
   prec = new IdentityInverse (a->SpMat());
   switch (iter_method)
   {
      case 1:
         inv = new PCGMatrixInverse (a->SpMat(), *prec, printit, maxit,
                                     rtol, atol);
         break;
      case 2:
         inv = new GMRESMatrixInverse(a->SpMat(), *prec, printit, maxit,
                                      itermethod_rstart, rtol, atol);
         break;
      case 3:
         inv = new BICGSTABMatrixInverse(a->SpMat(), *prec, printit,
                                         maxit, rtol, atol);
         break;
   }

   cout << "Solving the linear system ... " << endl;
   inv->Mult (*b, *sol);
   cout << "done solving." << endl;

   ::Vector *u = new ::Vector;
   sol->GetNodalValues (*u);

   delete inv;
   delete prec;
   delete sol;
   delete b;
   delete a;
   delete fespace;
   delete fec;
   delete mesh;

   TetVolField<double> *tvf =
      scinew TetVolField<double>(tvd->get_typed_mesh(), Field::NODE);
   tvf->resize_fdata();
   vector<double> &fdata = tvf->fdata();
   for (int i = 0; i < u->Size(); i++)
      fdata[i] = (*u)(i);

   delete u;

   outfieldhandle = tvf;

   // Send the data downstream.
   SimpleOPort<FieldHandle> *outport =
      (SimpleOPort<FieldHandle> *)getOPort("Field Output");
   if (!outport)
   {
      error("Unable to initialize 'Field Output' port.");
      return;
   }
   outport->send(outfieldhandle);
}

void FEM::tcl_command(GuiArgs& args, void* userdata)
{
   Module::tcl_command(args, userdata);
}

} // End namespace DDDAS


