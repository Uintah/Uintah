[errNorm,orders,success,tCPU,tElapsed,grid,A,b,x,TI] = testDisc;
x1 = loadHypreParVector('../HypreStandAlone/output_x1.par',4);
A1 = loadHypreIJMatrix('../HypreStandAlone/output_A.ij',4);
b1 = loadHypreParVector('../HypreStandAlone/output_b.par',4);
[success,A0,b0,x0] = compareSystems(grid,A,b,x,A1,b1,x1);