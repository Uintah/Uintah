ReadAndPlotTFailPrev <- function() {
  num_points = 50
  consistency_iter = 1
  eta_hi = 1
  eta_lo = 0
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72933e+06,8.35582e+06)
  z_r_yield_z = c(575.128,-251815,-912580,-1.72933e+06,-2.3901e+06,-2.64249e+06)
  z_r_yield_r = c(3.94984e-09,1.28184e+06,4.63773e+06,8.35582e+06,7.13672e+06,0)
  zr_df = 
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter)
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.09773e+06,8.62158e+06)
  z_r_yield_z = c(-1.32096e+06,-1.72933e+06,-2.09773e+06,-2.3901e+06,-2.57781e+06,-2.64249e+06)
  z_r_yield_r = c(6.71178e+06,8.35582e+06,8.62158e+06,7.13672e+06,4.04659e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-2.20523e+06,-2.3901e+06,-2.52823e+06,-2.61361e+06,-2.64249e+06)
  z_r_yield_r = c(8.71886e+06,8.32533e+06,7.13672e+06,5.22384e+06,2.76042e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-2.05455e+06,-2.12428e+06,-2.19064e+06,-2.25333e+06,-2.3121e+06)
  z_r_yield_r = c(8.71886e+06,8.68172e+06,8.56896e+06,8.37927e+06,8.11236e+06,7.76899e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-2.0161e+06,-2.04983e+06,-2.0829e+06,-2.11527e+06,-2.14691e+06)
  z_r_yield_r = c(8.71886e+06,8.7108e+06,8.68648e+06,8.64567e+06,8.58822e+06,8.51399e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-1.99854e+06,-2.01522e+06,-2.03174e+06,-2.04811e+06,-2.06432e+06)
  z_r_yield_r = c(8.71886e+06,8.71695e+06,8.71122e+06,8.70162e+06,8.68814e+06,8.67075e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-1.99005e+06,-1.99835e+06,-2.00661e+06,-2.01483e+06,-2.02302e+06)
  z_r_yield_r = c(8.71886e+06,8.71839e+06,8.71699e+06,8.71466e+06,8.71139e+06,8.70718e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-1.98587e+06,-1.99001e+06,-1.99414e+06,-1.99826e+06,-2.00237e+06)
  z_r_yield_r = c(8.71886e+06,8.71874e+06,8.7184e+06,8.71782e+06,8.71702e+06,8.71598e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-1.98379e+06,-1.98586e+06,-1.98792e+06,-1.98998e+06,-1.99204e+06)
  z_r_yield_r = c(8.71886e+06,8.71883e+06,8.71874e+06,8.7186e+06,8.7184e+06,8.71814e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-1.98275e+06,-1.98379e+06,-1.98482e+06,-1.98585e+06,-1.98688e+06)
  z_r_yield_r = c(8.71886e+06,8.71885e+06,8.71883e+06,8.71879e+06,8.71874e+06,8.71868e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.8513e+08
  pbar_w = 6.01844e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.98172e+06,8.71886e+06)
  z_r_yield_z = c(-1.98172e+06,-1.98224e+06,-1.98275e+06,-1.98327e+06,-1.98379e+06,-1.9843e+06)
  z_r_yield_r = c(8.71886e+06,8.71885e+06,8.71885e+06,8.71884e+06,8.71883e+06,8.71881e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 2
  eta_hi = 1
  eta_lo = 0.5
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72578e+06,8.33866e+06)
  z_r_yield_z = c(575.128,-251296,-910704,-1.72578e+06,-2.38518e+06,-2.63705e+06)
  z_r_yield_r = c(3.94984e-09,1.2792e+06,4.6282e+06,8.33866e+06,7.12206e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.09342e+06,8.60387e+06)
  z_r_yield_z = c(-1.31824e+06,-1.72578e+06,-2.09342e+06,-2.38518e+06,-2.57251e+06,-2.63705e+06)
  z_r_yield_r = c(6.69799e+06,8.33866e+06,8.60387e+06,7.12206e+06,4.03828e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-2.2007e+06,-2.38518e+06,-2.52304e+06,-2.60824e+06,-2.63705e+06)
  z_r_yield_r = c(8.70094e+06,8.30822e+06,7.12206e+06,5.2131e+06,2.75475e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-2.05032e+06,-2.11992e+06,-2.18614e+06,-2.2487e+06,-2.30735e+06)
  z_r_yield_r = c(8.70094e+06,8.66388e+06,8.55135e+06,8.36205e+06,8.0957e+06,7.75303e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-2.01195e+06,-2.04562e+06,-2.07862e+06,-2.11092e+06,-2.1425e+06)
  z_r_yield_r = c(8.70094e+06,8.6929e+06,8.66863e+06,8.62791e+06,8.57057e+06,8.4965e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-1.99443e+06,-2.01107e+06,-2.02756e+06,-2.0439e+06,-2.06007e+06)
  z_r_yield_r = c(8.70094e+06,8.69904e+06,8.69332e+06,8.68374e+06,8.67029e+06,8.65293e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-1.98596e+06,-1.99424e+06,-2.00248e+06,-2.01069e+06,-2.01886e+06)
  z_r_yield_r = c(8.70094e+06,8.70048e+06,8.69908e+06,8.69676e+06,8.69349e+06,8.68929e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-1.98179e+06,-1.98592e+06,-1.99004e+06,-1.99415e+06,-1.99825e+06)
  z_r_yield_r = c(8.70094e+06,8.70083e+06,8.70048e+06,8.69991e+06,8.6991e+06,8.69807e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-1.97971e+06,-1.98178e+06,-1.98384e+06,-1.98589e+06,-1.98795e+06)
  z_r_yield_r = c(8.70094e+06,8.70091e+06,8.70083e+06,8.70068e+06,8.70049e+06,8.70023e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-1.97868e+06,-1.97971e+06,-1.98074e+06,-1.98177e+06,-1.9828e+06)
  z_r_yield_r = c(8.70094e+06,8.70093e+06,8.70091e+06,8.70088e+06,8.70083e+06,8.70076e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.84224e+08
  pbar_w = 5.98855e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97765e+06,8.70094e+06)
  z_r_yield_z = c(-1.97765e+06,-1.97816e+06,-1.97868e+06,-1.97919e+06,-1.97971e+06,-1.98022e+06)
  z_r_yield_r = c(8.70094e+06,8.70094e+06,8.70093e+06,8.70093e+06,8.70091e+06,8.7009e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 3
  eta_hi = 1
  eta_lo = 0.75
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.724e+06,8.33008e+06)
  z_r_yield_z = c(575.128,-251037,-909766,-1.724e+06,-2.38273e+06,-2.63434e+06)
  z_r_yield_r = c(3.94984e-09,1.27789e+06,4.62344e+06,8.33008e+06,7.11474e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.09127e+06,8.59502e+06)
  z_r_yield_z = c(-1.31688e+06,-1.724e+06,-2.09127e+06,-2.38273e+06,-2.56986e+06,-2.63434e+06)
  z_r_yield_r = c(6.6911e+06,8.33008e+06,8.59502e+06,7.11474e+06,4.03413e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-2.19844e+06,-2.38273e+06,-2.52044e+06,-2.60555e+06,-2.63434e+06)
  z_r_yield_r = c(8.69199e+06,8.29968e+06,7.11474e+06,5.20774e+06,2.75191e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-2.04821e+06,-2.11774e+06,-2.18389e+06,-2.24639e+06,-2.30498e+06)
  z_r_yield_r = c(8.69199e+06,8.65497e+06,8.54256e+06,8.35345e+06,8.08737e+06,7.74505e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-2.00988e+06,-2.04352e+06,-2.07648e+06,-2.10875e+06,-2.1403e+06)
  z_r_yield_r = c(8.69199e+06,8.68396e+06,8.65972e+06,8.61903e+06,8.56176e+06,8.48776e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-1.99238e+06,-2.00901e+06,-2.02548e+06,-2.04179e+06,-2.05795e+06)
  z_r_yield_r = c(8.69199e+06,8.69009e+06,8.68438e+06,8.67481e+06,8.66137e+06,8.64404e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-1.98392e+06,-1.99219e+06,-2.00042e+06,-2.00862e+06,-2.01678e+06)
  z_r_yield_r = c(8.69199e+06,8.69153e+06,8.69014e+06,8.68781e+06,8.68455e+06,8.68035e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-1.97975e+06,-1.98387e+06,-1.98799e+06,-1.9921e+06,-1.9962e+06)
  z_r_yield_r = c(8.69199e+06,8.69188e+06,8.69153e+06,8.69096e+06,8.69016e+06,8.68912e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-1.97768e+06,-1.97974e+06,-1.98179e+06,-1.98385e+06,-1.98591e+06)
  z_r_yield_r = c(8.69199e+06,8.69196e+06,8.69188e+06,8.69174e+06,8.69154e+06,8.69128e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-1.97664e+06,-1.97767e+06,-1.9787e+06,-1.97973e+06,-1.98076e+06)
  z_r_yield_r = c(8.69199e+06,8.69199e+06,8.69196e+06,8.69193e+06,8.69188e+06,8.69182e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83771e+08
  pbar_w = 5.97361e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97561e+06,8.69199e+06)
  z_r_yield_z = c(-1.97561e+06,-1.97613e+06,-1.97664e+06,-1.97716e+06,-1.97767e+06,-1.97819e+06)
  z_r_yield_r = c(8.69199e+06,8.69199e+06,8.69199e+06,8.69198e+06,8.69196e+06,8.69195e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 4
  eta_hi = 1
  eta_lo = 0.875
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72311e+06,8.32579e+06)
  z_r_yield_z = c(575.128,-250908,-909298,-1.72311e+06,-2.3815e+06,-2.63299e+06)
  z_r_yield_r = c(3.94984e-09,1.27723e+06,4.62106e+06,8.32579e+06,7.11107e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.09019e+06,8.5906e+06)
  z_r_yield_z = c(-1.31621e+06,-1.72311e+06,-2.09019e+06,-2.3815e+06,-2.56854e+06,-2.63299e+06)
  z_r_yield_r = c(6.68766e+06,8.32579e+06,8.5906e+06,7.11107e+06,4.03205e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-2.1973e+06,-2.3815e+06,-2.51914e+06,-2.60421e+06,-2.63299e+06)
  z_r_yield_r = c(8.68752e+06,8.29541e+06,7.11107e+06,5.20506e+06,2.7505e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-2.04716e+06,-2.11665e+06,-2.18276e+06,-2.24523e+06,-2.30379e+06)
  z_r_yield_r = c(8.68752e+06,8.65052e+06,8.53816e+06,8.34915e+06,8.08321e+06,7.74107e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-2.00885e+06,-2.04247e+06,-2.07542e+06,-2.10767e+06,-2.13919e+06)
  z_r_yield_r = c(8.68752e+06,8.6795e+06,8.65526e+06,8.6146e+06,8.55735e+06,8.48339e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-1.99136e+06,-2.00797e+06,-2.02443e+06,-2.04074e+06,-2.05689e+06)
  z_r_yield_r = c(8.68752e+06,8.68562e+06,8.67991e+06,8.67035e+06,8.65691e+06,8.63959e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-1.9829e+06,-1.99116e+06,-1.99939e+06,-2.00759e+06,-2.01575e+06)
  z_r_yield_r = c(8.68752e+06,8.68706e+06,8.68567e+06,8.68334e+06,8.68008e+06,8.67588e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-1.97873e+06,-1.98285e+06,-1.98697e+06,-1.99107e+06,-1.99517e+06)
  z_r_yield_r = c(8.68752e+06,8.68741e+06,8.68706e+06,8.68649e+06,8.68569e+06,8.68465e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-1.97666e+06,-1.97872e+06,-1.98077e+06,-1.98283e+06,-1.98488e+06)
  z_r_yield_r = c(8.68752e+06,8.68749e+06,8.68741e+06,8.68726e+06,8.68707e+06,8.68681e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-1.97563e+06,-1.97666e+06,-1.97768e+06,-1.97871e+06,-1.97974e+06)
  z_r_yield_r = c(8.68752e+06,8.68751e+06,8.68749e+06,8.68746e+06,8.68741e+06,8.68734e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83545e+08
  pbar_w = 5.96614e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9746e+06,8.68752e+06)
  z_r_yield_z = c(-1.9746e+06,-1.97511e+06,-1.97563e+06,-1.97614e+06,-1.97665e+06,-1.97717e+06)
  z_r_yield_r = c(8.68752e+06,8.68752e+06,8.68751e+06,8.6875e+06,8.68749e+06,8.68748e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 5
  eta_hi = 1
  eta_lo = 0.9375
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72267e+06,8.32365e+06)
  z_r_yield_z = c(575.128,-250843,-909064,-1.72267e+06,-2.38089e+06,-2.63231e+06)
  z_r_yield_r = c(3.94984e-09,1.2769e+06,4.61987e+06,8.32365e+06,7.10924e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.08965e+06,8.58839e+06)
  z_r_yield_z = c(-1.31587e+06,-1.72267e+06,-2.08965e+06,-2.38089e+06,-2.56788e+06,-2.63231e+06)
  z_r_yield_r = c(6.68594e+06,8.32365e+06,8.58839e+06,7.10924e+06,4.03101e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-2.19674e+06,-2.38089e+06,-2.5185e+06,-2.60354e+06,-2.63231e+06)
  z_r_yield_r = c(8.68529e+06,8.29327e+06,7.10924e+06,5.20372e+06,2.74979e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-2.04663e+06,-2.1161e+06,-2.1822e+06,-2.24465e+06,-2.3032e+06)
  z_r_yield_r = c(8.68529e+06,8.64829e+06,8.53596e+06,8.347e+06,8.08113e+06,7.73908e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-2.00833e+06,-2.04194e+06,-2.07488e+06,-2.10713e+06,-2.13864e+06)
  z_r_yield_r = c(8.68529e+06,8.67726e+06,8.65303e+06,8.61238e+06,8.55515e+06,8.48121e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-1.99084e+06,-2.00745e+06,-2.02391e+06,-2.04022e+06,-2.05637e+06)
  z_r_yield_r = c(8.68529e+06,8.68339e+06,8.67767e+06,8.66812e+06,8.65469e+06,8.63736e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-1.98239e+06,-1.99065e+06,-1.99888e+06,-2.00707e+06,-2.01523e+06)
  z_r_yield_r = c(8.68529e+06,8.68482e+06,8.68343e+06,8.68111e+06,8.67785e+06,8.67365e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-1.97822e+06,-1.98234e+06,-1.98646e+06,-1.99056e+06,-1.99466e+06)
  z_r_yield_r = c(8.68529e+06,8.68517e+06,8.68483e+06,8.68425e+06,8.68345e+06,8.68242e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-1.97615e+06,-1.97821e+06,-1.98027e+06,-1.98232e+06,-1.98437e+06)
  z_r_yield_r = c(8.68529e+06,8.68526e+06,8.68517e+06,8.68503e+06,8.68483e+06,8.68457e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-1.97512e+06,-1.97615e+06,-1.97717e+06,-1.9782e+06,-1.97923e+06)
  z_r_yield_r = c(8.68529e+06,8.68528e+06,8.68526e+06,8.68522e+06,8.68517e+06,8.68511e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83431e+08
  pbar_w = 5.9624e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97409e+06,8.68529e+06)
  z_r_yield_z = c(-1.97409e+06,-1.9746e+06,-1.97512e+06,-1.97563e+06,-1.97614e+06,-1.97666e+06)
  z_r_yield_r = c(8.68529e+06,8.68528e+06,8.68528e+06,8.68527e+06,8.68526e+06,8.68524e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 6
  eta_hi = 1
  eta_lo = 0.96875
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72245e+06,8.32258e+06)
  z_r_yield_z = c(575.128,-250811,-908947,-1.72245e+06,-2.38058e+06,-2.63197e+06)
  z_r_yield_r = c(3.94984e-09,1.27674e+06,4.61928e+06,8.32258e+06,7.10833e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.08938e+06,8.58728e+06)
  z_r_yield_z = c(-1.3157e+06,-1.72245e+06,-2.08938e+06,-2.38058e+06,-2.56755e+06,-2.63197e+06)
  z_r_yield_r = c(6.68508e+06,8.32258e+06,8.58728e+06,7.10833e+06,4.0305e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-2.19646e+06,-2.38058e+06,-2.51817e+06,-2.60321e+06,-2.63197e+06)
  z_r_yield_r = c(8.68417e+06,8.29221e+06,7.10833e+06,5.20305e+06,2.74944e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-2.04637e+06,-2.11583e+06,-2.18192e+06,-2.24437e+06,-2.3029e+06)
  z_r_yield_r = c(8.68417e+06,8.64718e+06,8.53487e+06,8.34593e+06,8.08009e+06,7.73808e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-2.00807e+06,-2.04168e+06,-2.07461e+06,-2.10685e+06,-2.13837e+06)
  z_r_yield_r = c(8.68417e+06,8.67615e+06,8.65192e+06,8.61128e+06,8.55405e+06,8.48012e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-1.99059e+06,-2.0072e+06,-2.02365e+06,-2.03996e+06,-2.0561e+06)
  z_r_yield_r = c(8.68417e+06,8.68227e+06,8.67656e+06,8.667e+06,8.65357e+06,8.63625e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-1.98213e+06,-1.9904e+06,-1.99862e+06,-2.00681e+06,-2.01497e+06)
  z_r_yield_r = c(8.68417e+06,8.6837e+06,8.68231e+06,8.67999e+06,8.67673e+06,8.67254e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-1.97796e+06,-1.98209e+06,-1.9862e+06,-1.9903e+06,-1.9944e+06)
  z_r_yield_r = c(8.68417e+06,8.68405e+06,8.68371e+06,8.68314e+06,8.68233e+06,8.6813e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-1.97589e+06,-1.97795e+06,-1.98001e+06,-1.98206e+06,-1.98412e+06)
  z_r_yield_r = c(8.68417e+06,8.68414e+06,8.68405e+06,8.68391e+06,8.68371e+06,8.68346e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-1.97486e+06,-1.97589e+06,-1.97692e+06,-1.97795e+06,-1.97898e+06)
  z_r_yield_r = c(8.68417e+06,8.68416e+06,8.68414e+06,8.6841e+06,8.68405e+06,8.68399e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83375e+08
  pbar_w = 5.96054e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97383e+06,8.68417e+06)
  z_r_yield_z = c(-1.97383e+06,-1.97435e+06,-1.97486e+06,-1.97538e+06,-1.97589e+06,-1.9764e+06)
  z_r_yield_r = c(8.68417e+06,8.68417e+06,8.68416e+06,8.68415e+06,8.68414e+06,8.68412e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 7
  eta_hi = 1
  eta_lo = 0.984375
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72234e+06,8.32204e+06)
  z_r_yield_z = c(575.128,-250794,-908888,-1.72234e+06,-2.38043e+06,-2.6318e+06)
  z_r_yield_r = c(3.94984e-09,1.27665e+06,4.61898e+06,8.32204e+06,7.10787e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.08925e+06,8.58673e+06)
  z_r_yield_z = c(-1.31561e+06,-1.72234e+06,-2.08925e+06,-2.38043e+06,-2.56738e+06,-2.6318e+06)
  z_r_yield_r = c(6.68465e+06,8.32204e+06,8.58673e+06,7.10787e+06,4.03024e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-2.19631e+06,-2.38043e+06,-2.51801e+06,-2.60304e+06,-2.6318e+06)
  z_r_yield_r = c(8.68361e+06,8.29167e+06,7.10787e+06,5.20272e+06,2.74926e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-2.04624e+06,-2.11569e+06,-2.18178e+06,-2.24422e+06,-2.30275e+06)
  z_r_yield_r = c(8.68361e+06,8.64662e+06,8.53432e+06,8.34539e+06,8.07957e+06,7.73758e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-2.00794e+06,-2.04155e+06,-2.07448e+06,-2.10672e+06,-2.13823e+06)
  z_r_yield_r = c(8.68361e+06,8.67559e+06,8.65136e+06,8.61072e+06,8.5535e+06,8.47957e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-1.99046e+06,-2.00707e+06,-2.02352e+06,-2.03982e+06,-2.05597e+06)
  z_r_yield_r = c(8.68361e+06,8.68171e+06,8.676e+06,8.66644e+06,8.65302e+06,8.6357e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-1.982e+06,-1.99027e+06,-1.99849e+06,-2.00668e+06,-2.01484e+06)
  z_r_yield_r = c(8.68361e+06,8.68315e+06,8.68176e+06,8.67943e+06,8.67617e+06,8.67198e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-1.97784e+06,-1.98196e+06,-1.98607e+06,-1.99018e+06,-1.99427e+06)
  z_r_yield_r = c(8.68361e+06,8.68349e+06,8.68315e+06,8.68258e+06,8.68178e+06,8.68074e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-1.97577e+06,-1.97783e+06,-1.97988e+06,-1.98194e+06,-1.98399e+06)
  z_r_yield_r = c(8.68361e+06,8.68358e+06,8.6835e+06,8.68335e+06,8.68315e+06,8.6829e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-1.97474e+06,-1.97576e+06,-1.97679e+06,-1.97782e+06,-1.97885e+06)
  z_r_yield_r = c(8.68361e+06,8.6836e+06,8.68358e+06,8.68355e+06,8.6835e+06,8.68343e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83346e+08
  pbar_w = 5.9596e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97371e+06,8.68361e+06)
  z_r_yield_z = c(-1.97371e+06,-1.97422e+06,-1.97474e+06,-1.97525e+06,-1.97576e+06,-1.97628e+06)
  z_r_yield_r = c(8.68361e+06,8.68361e+06,8.6836e+06,8.68359e+06,8.68358e+06,8.68356e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 8
  eta_hi = 1
  eta_lo = 0.992188
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72228e+06,8.32178e+06)
  z_r_yield_z = c(575.128,-250786,-908859,-1.72228e+06,-2.38035e+06,-2.63172e+06)
  z_r_yield_r = c(3.94984e-09,1.27661e+06,4.61883e+06,8.32178e+06,7.10764e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.08918e+06,8.58645e+06)
  z_r_yield_z = c(-1.31557e+06,-1.72228e+06,-2.08918e+06,-2.38035e+06,-2.5673e+06,-2.63172e+06)
  z_r_yield_r = c(6.68443e+06,8.32178e+06,8.58645e+06,7.10764e+06,4.03011e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-2.19624e+06,-2.38035e+06,-2.51793e+06,-2.60295e+06,-2.63172e+06)
  z_r_yield_r = c(8.68333e+06,8.29141e+06,7.10764e+06,5.20255e+06,2.74917e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-2.04617e+06,-2.11562e+06,-2.18171e+06,-2.24415e+06,-2.30268e+06)
  z_r_yield_r = c(8.68333e+06,8.64635e+06,8.53404e+06,8.34513e+06,8.07931e+06,7.73734e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-2.00788e+06,-2.04148e+06,-2.07441e+06,-2.10665e+06,-2.13816e+06)
  z_r_yield_r = c(8.68333e+06,8.67531e+06,8.65108e+06,8.61044e+06,8.55322e+06,8.4793e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-1.9904e+06,-2.007e+06,-2.02346e+06,-2.03976e+06,-2.0559e+06)
  z_r_yield_r = c(8.68333e+06,8.68143e+06,8.67572e+06,8.66616e+06,8.65274e+06,8.63542e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-1.98194e+06,-1.9902e+06,-1.99843e+06,-2.00662e+06,-2.01477e+06)
  z_r_yield_r = c(8.68333e+06,8.68287e+06,8.68148e+06,8.67915e+06,8.6759e+06,8.6717e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-1.97777e+06,-1.9819e+06,-1.98601e+06,-1.99011e+06,-1.99421e+06)
  z_r_yield_r = c(8.68333e+06,8.68322e+06,8.68287e+06,8.6823e+06,8.6815e+06,8.68046e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-1.9757e+06,-1.97776e+06,-1.97982e+06,-1.98187e+06,-1.98393e+06)
  z_r_yield_r = c(8.68333e+06,8.6833e+06,8.68322e+06,8.68307e+06,8.68287e+06,8.68262e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-1.97467e+06,-1.9757e+06,-1.97673e+06,-1.97776e+06,-1.97878e+06)
  z_r_yield_r = c(8.68333e+06,8.68332e+06,8.6833e+06,8.68327e+06,8.68322e+06,8.68315e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83332e+08
  pbar_w = 5.95913e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97364e+06,8.68333e+06)
  z_r_yield_z = c(-1.97364e+06,-1.97416e+06,-1.97467e+06,-1.97519e+06,-1.9757e+06,-1.97621e+06)
  z_r_yield_r = c(8.68333e+06,8.68333e+06,8.68332e+06,8.68331e+06,8.6833e+06,8.68329e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 9
  eta_hi = 1
  eta_lo = 0.996094
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72225e+06,8.32164e+06)
  z_r_yield_z = c(575.128,-250782,-908844,-1.72225e+06,-2.38032e+06,-2.63167e+06)
  z_r_yield_r = c(3.94984e-09,1.27659e+06,4.61876e+06,8.32164e+06,7.10753e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.08915e+06,8.58631e+06)
  z_r_yield_z = c(-1.31555e+06,-1.72225e+06,-2.08915e+06,-2.38032e+06,-2.56726e+06,-2.63167e+06)
  z_r_yield_r = c(6.68432e+06,8.32164e+06,8.58631e+06,7.10753e+06,4.03004e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-2.19621e+06,-2.38032e+06,-2.51789e+06,-2.60291e+06,-2.63167e+06)
  z_r_yield_r = c(8.68319e+06,8.29127e+06,7.10753e+06,5.20247e+06,2.74913e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-2.04614e+06,-2.11559e+06,-2.18167e+06,-2.24411e+06,-2.30264e+06)
  z_r_yield_r = c(8.68319e+06,8.64621e+06,8.5339e+06,8.34499e+06,8.07918e+06,7.73721e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-2.00785e+06,-2.04145e+06,-2.07438e+06,-2.10662e+06,-2.13813e+06)
  z_r_yield_r = c(8.68319e+06,8.67517e+06,8.65095e+06,8.61031e+06,8.55308e+06,8.47916e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-1.99036e+06,-2.00697e+06,-2.02343e+06,-2.03973e+06,-2.05587e+06)
  z_r_yield_r = c(8.68319e+06,8.68129e+06,8.67558e+06,8.66602e+06,8.6526e+06,8.63528e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-1.98191e+06,-1.99017e+06,-1.9984e+06,-2.00659e+06,-2.01474e+06)
  z_r_yield_r = c(8.68319e+06,8.68273e+06,8.68134e+06,8.67901e+06,8.67576e+06,8.67156e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-1.97774e+06,-1.98186e+06,-1.98598e+06,-1.99008e+06,-1.99418e+06)
  z_r_yield_r = c(8.68319e+06,8.68308e+06,8.68273e+06,8.68216e+06,8.68136e+06,8.68032e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-1.97567e+06,-1.97773e+06,-1.97979e+06,-1.98184e+06,-1.98389e+06)
  z_r_yield_r = c(8.68319e+06,8.68316e+06,8.68308e+06,8.68293e+06,8.68273e+06,8.68248e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-1.97464e+06,-1.97567e+06,-1.9767e+06,-1.97773e+06,-1.97875e+06)
  z_r_yield_r = c(8.68319e+06,8.68318e+06,8.68316e+06,8.68313e+06,8.68308e+06,8.68301e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83325e+08
  pbar_w = 5.9589e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.97361e+06,8.68319e+06)
  z_r_yield_z = c(-1.97361e+06,-1.97413e+06,-1.97464e+06,-1.97515e+06,-1.97567e+06,-1.97618e+06)
  z_r_yield_r = c(8.68319e+06,8.68319e+06,8.68318e+06,8.68317e+06,8.68316e+06,8.68315e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  consistency_iter = 10
  eta_hi = 1
  eta_lo = 0.998047
  iteration = 1
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.72224e+06,8.32158e+06)
  z_r_yield_z = c(575.128,-250780,-908837,-1.72224e+06,-2.3803e+06,-2.63165e+06)
  z_r_yield_r = c(3.94984e-09,1.27658e+06,4.61872e+06,8.32158e+06,7.10747e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 2
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-2.08913e+06,8.58625e+06)
  z_r_yield_z = c(-1.31554e+06,-1.72224e+06,-2.08913e+06,-2.3803e+06,-2.56724e+06,-2.63165e+06)
  z_r_yield_r = c(6.68427e+06,8.32158e+06,8.58625e+06,7.10747e+06,4.03001e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-2.19619e+06,-2.3803e+06,-2.51787e+06,-2.60289e+06,-2.63165e+06)
  z_r_yield_r = c(8.68312e+06,8.29121e+06,7.10747e+06,5.20243e+06,2.74911e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-2.04612e+06,-2.11557e+06,-2.18166e+06,-2.2441e+06,-2.30262e+06)
  z_r_yield_r = c(8.68312e+06,8.64614e+06,8.53384e+06,8.34492e+06,8.07911e+06,7.73715e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-2.00783e+06,-2.04143e+06,-2.07436e+06,-2.1066e+06,-2.13811e+06)
  z_r_yield_r = c(8.68312e+06,8.6751e+06,8.65088e+06,8.61024e+06,8.55302e+06,8.4791e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-1.99035e+06,-2.00695e+06,-2.02341e+06,-2.03971e+06,-2.05585e+06)
  z_r_yield_r = c(8.68312e+06,8.68122e+06,8.67551e+06,8.66595e+06,8.65253e+06,8.63521e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-1.98189e+06,-1.99015e+06,-1.99838e+06,-2.00657e+06,-2.01472e+06)
  z_r_yield_r = c(8.68312e+06,8.68266e+06,8.68127e+06,8.67894e+06,8.67569e+06,8.67149e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-1.97773e+06,-1.98185e+06,-1.98596e+06,-1.99006e+06,-1.99416e+06)
  z_r_yield_r = c(8.68312e+06,8.68301e+06,8.68266e+06,8.68209e+06,8.68129e+06,8.68025e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-1.97566e+06,-1.97771e+06,-1.97977e+06,-1.98183e+06,-1.98388e+06)
  z_r_yield_r = c(8.68312e+06,8.68309e+06,8.68301e+06,8.68286e+06,8.68266e+06,8.68241e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-1.97462e+06,-1.97565e+06,-1.97668e+06,-1.97771e+06,-1.97874e+06)
  z_r_yield_r = c(8.68312e+06,8.68311e+06,8.68309e+06,8.68306e+06,8.68301e+06,8.68294e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.29199e+09
  G = 2.33113e+08
  X = -1.83322e+08
  pbar_w = 5.95878e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355315, PEAKI1 = 996.151, STREN = 1.76387e+07, YSLOPE = 0.353634)
  z_r_pt = c(169338,3.03967e+07)
  z_r_closest = c(-1.9736e+06,8.68312e+06)
  z_r_yield_z = c(-1.9736e+06,-1.97411e+06,-1.97462e+06,-1.97514e+06,-1.97565e+06,-1.97617e+06)
  z_r_yield_r = c(8.68312e+06,8.68312e+06,8.68311e+06,8.6831e+06,8.68309e+06,8.68308e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))

  iteration = 10
  createGIFConsistency(zr_df, iteration)

  return (zr_df)
}
