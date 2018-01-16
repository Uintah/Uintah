require("ggplot2")
require("animation")
require("latex2exp")

#------------------------------------------------------
# Extract a single iteration
#------------------------------------------------------
extractIteration <- function(yieldSurfData, iteration, consistency_iter) {

  print(paste("iteration", iteration, "consistency_iter = ", consistency_iter))

  return(yieldSurfData[which(yieldSurfData$Iteration == iteration &
                             yieldSurfData$CIteration == consistency_iter),])
}

#------------------------------------------------------
# Plot single iteration
#------------------------------------------------------
plotYieldSurface <- function(zrprime_data) {

  zrprime_trial_closest = zrprime_data[which(zrprime_data$Label == "trial" |
                                             zrprime_data$Label == "closest"),]
  plt = ggplot() + 
        geom_path(data = zrprime_data, 
                  aes(x = z*1.0e-6, y = rprime*1.0e-6, group=Label, color = Label),
                      size = 1)+
        geom_point(data = zrprime_data, 
                   aes(x = z*1.0e-6, y = rprime*1.0e-6, group=Label, color = Label)) +
        geom_line(data = zrprime_trial_closest,
                  aes(x = z*1.0e-6, y = rprime*1.0e-6),
                  color = "red", linetype = 1,
                  size = 1) +
        xlab(TeX(paste("$z = I_1/\\sqrt{3}$", "MPa"))) +
        ylab(TeX(paste("$r' = \\beta\\sqrt{3K/2G}\\sqrt{2J_2}$", "MPa"))) +
        #coord_fixed() +
        theme_bw() + 
        theme(legend.justification=c(0,1), legend.position=c(0,1),
            plot.title = element_text(size = 14),
            axis.title.x = element_text(size=16),
            axis.title.y = element_text(size=16),
            axis.text.x = element_text(size=14),
            axis.text.y = element_text(size=14),
            legend.text = element_text(size=12))
  #dev.new()
  print(plt)
}

#------------------------------------------------------
# Animate the plots
#------------------------------------------------------
animateIterations <- function(yieldSurfData, num_iterations, consistency_iter) {
  lapply(seq(1, num_iterations, 1), 
         function(iteration) {
           plotYieldSurface(extractIteration(yieldSurfData, iteration, consistency_iter))
         })
}
animateConsistency <- function(yieldSurfData, iteration, num_consistency_iter) {
  lapply(seq(1, num_consistency_iter, 1), 
         function(consistency_iter) {
           print(paste("iteration = ", iteration, "consistency_iter = ", consistency_iter))
           plotYieldSurface(extractIteration(yieldSurfData, iteration, consistency_iter))
         })
}

#------------------------------------------------------
# Function to create animated gif
#------------------------------------------------------
createGIFIterations <- function(yieldSurfData, consistency_iter) {

  outputGIFFile = paste0("test_iter", ".gif")
  print(outputGIFFile)

  # Compute number of iterations
  num_iterations = length(unique(yieldSurfData$Iteration))

  # Save as animation
  ani.options(ani.height = 600, ani.width = 600)
  saveGIF(animateIterations(yieldSurfData, num_iterations, consistency_iter), 
          interval=1.0, 
          movie.name=outputGIFFile)
}

createGIFConsistency <- function(yieldSurfData, iteration) {

  outputGIFFile = paste0("test_consis", ".gif")
  print(outputGIFFile)

  # Compute number of consistency iterations
  num_consistency_iter = length(unique(yieldSurfData$CIteration))
  print(paste("num_consistency_iter = ", num_consistency_iter))

  # Save as animation
  saveGIF(animateConsistency(yieldSurfData, iteration, num_consistency_iter), 
          interval=0.5, 
          movie.name=outputGIFFile)
}

#-------------------------------------------------------------------------
# Compute the full yield surface and create data frame
#-------------------------------------------------------------------------
ComputeFullYieldSurface <- function(yieldParams, capX, pbar_w, K, G, num_points,
                                    z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                                    iteration, consistency_iter) {

  # Get yield parameters
  yieldParams = unlist(yieldParams)
  PEAKI1 = as.numeric(yieldParams['PEAKI1'])
  FSLOPE = as.numeric(yieldParams['FSLOPE'])
  STREN  = as.numeric(yieldParams['STREN'])
  YSLOPE = as.numeric(yieldParams['YSLOPE'])
  CR     = as.numeric(yieldParams['CR'])

  # Set up constants
  a1 = STREN
  a2 = (FSLOPE-YSLOPE)/(STREN-YSLOPE*PEAKI1)
  a3 = (STREN-YSLOPE*PEAKI1)*exp(-a2*PEAKI1)
  a4 = YSLOPE

  # Compute kappa
  X_eff = capX + 3*pbar_w
  kappa = PEAKI1 - CR*(PEAKI1 - X_eff)

  # Create an array of I1_eff values
  I1eff_min = X_eff;
  I1eff_max = PEAKI1;

  rad = 0.5*(PEAKI1 - X_eff)
  cen = 0.5*(PEAKI1 + X_eff)
  theta_max = acos(max((I1eff_min - cen)/rad, -1.0))
  theta_min = acos(min((I1eff_max - cen)/rad, 1.0))
  theta_vec = seq(from = theta_min, to = theta_max, length.out = num_points)
  I1_list = cen + rad*cos(theta_vec)
  I1_eff_list = lapply(I1_list, function(val) {max(val, X_eff)})
  sqrtJ2_list = lapply(I1_eff_list,
    function(I1_eff) {
      # Compute F_f
      Ff = a1 - a3*exp(a2*I1_eff) - a4*(I1_eff)

      # Compute Fc
      Fc_sq = 1.0
      if ((I1_eff < kappa) && (X_eff <= I1_eff)) {
        ratio = (kappa - I1_eff)/(kappa - X_eff)
        Fc_sq = 1.0 - ratio^2
      }

      # Compute sqrt(J2)
      sqrtJ2 = Ff*sqrt(Fc_sq)
      return(sqrtJ2)
    })
  print(paste("consistency_iter = ", consistency_iter))
  zrprime_data = data.frame(z = unlist(I1_eff_list)/sqrt(3), 
                            rprime = unlist(sqrtJ2_list)*sqrt(2)*sqrt(1.5*K/G), 
                            Iteration = as.factor(iteration),
                            CIteration = as.factor(consistency_iter),
                            Label="yield")
  zrprime_data = rbind(zrprime_data,
                       data.frame(z = z_r_pt[1], rprime = z_r_pt[2], 
                                  Iteration = as.factor(iteration),
                                  CIteration = as.factor(consistency_iter),
                                  Label = "trial"))
  zrprime_data = rbind(zrprime_data,
                       data.frame(z = z_r_closest[1], rprime = z_r_closest[2], 
                                  Iteration = as.factor(iteration),
                                  CIteration = as.factor(consistency_iter),
                                  Label = "closest"))
  zrprime_data = rbind(zrprime_data,
                       data.frame(z = z_r_yield_z, rprime = z_r_yield_r, 
                                  Iteration = as.factor(iteration),
                                  CIteration = as.factor(consistency_iter),
                                  Label = "polyline"))
 
  return(zrprime_data)
}

#-------------------------------------------------------------------------
# Plot the full yield surface
#-------------------------------------------------------------------------
ComputeAndPlotFullYieldSurface <- function(yieldParams, capX, pbar_w, K, G, num_points,
                                           type = "I1J2",
                                           z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r) {

  # Get yield parameters
  yieldParams = unlist(yieldParams)
  PEAKI1 = as.numeric(yieldParams['PEAKI1'])
  FSLOPE = as.numeric(yieldParams['FSLOPE'])
  STREN  = as.numeric(yieldParams['STREN'])
  YSLOPE = as.numeric(yieldParams['YSLOPE'])
  CR     = as.numeric(yieldParams['CR'])

  # Set up constants
  a1 = STREN
  a2 = (FSLOPE-YSLOPE)/(STREN-YSLOPE*PEAKI1)
  a3 = (STREN-YSLOPE*PEAKI1)*exp(-a2*PEAKI1)
  a4 = YSLOPE

  # Compute kappa
  X_eff = capX + 3*pbar_w
  kappa = PEAKI1 - CR*(PEAKI1 - X_eff)

  # Create an array of I1_eff values
  I1eff_min = X_eff;
  I1eff_max = PEAKI1;

  rad = 0.5*(PEAKI1 - X_eff)
  cen = 0.5*(PEAKI1 + X_eff)
  theta_max = acos(max((I1eff_min - cen)/rad, -1.0))
  theta_min = acos(min((I1eff_max - cen)/rad, 1.0))
  theta_vec = seq(from = theta_min, to = theta_max, length.out = num_points)
  #print(paste("theta_max = ", theta_max, "theta_min = ", theta_min))
  #print(paste("theta_vec = "))
  #print(theta_vec)
  I1_list = cen + rad*cos(theta_vec)
  I1_eff_list = lapply(I1_list, function(val) {max(val, X_eff)})
  #print(paste("I1_eff_list = "))
  #print(unlist(I1_eff_list))
  sqrtJ2_list = lapply(I1_eff_list,
    function(I1_eff) {
      # Compute F_f
      Ff = a1 - a3*exp(a2*I1_eff) - a4*(I1_eff)

      # Compute Fc
      Fc_sq = 1.0
      if ((I1_eff < kappa) && (X_eff <= I1_eff)) {
        ratio = (kappa - I1_eff)/(kappa - X_eff)
        Fc_sq = 1.0 - ratio^2
      }

      # Compute sqrt(J2)
      sqrtJ2 = Ff*sqrt(Fc_sq)
      #print(paste("Ff = ", Ff, "Fc_sq = ", Fc_sq))
      return(sqrtJ2)
    })
  #print(paste("sqrtJ2_list = "))
  #print(unlist(sqrtJ2_list))
 
  # Plot the yield surface
  if (type == "I1J2") {
    I1J2_data = data.frame(I1_eff = unlist(I1_eff_list), 
                           sqrtJ2 = unlist(sqrtJ2_list))
    print(head(I1J2_data))
    plt = ggplot(data = I1J2_data) +
          geom_path(aes(x = I1_eff, y = sqrtJ2))
    print(plt)
  } else {
    zrprime_data = data.frame(z = unlist(I1_eff_list)/sqrt(3), 
                              rprime = unlist(sqrtJ2_list)*sqrt(2)*sqrt(1.5*K/G), Label="yield")
    zrprime_data = rbind(zrprime_data,
                         data.frame(z = z_r_pt[1], rprime = z_r_pt[2], Label = "trial"))
    zrprime_data = rbind(zrprime_data,
                         data.frame(z = z_r_closest[1], rprime = z_r_closest[2], Label = "closest"))
    zrprime_data = rbind(zrprime_data,
                         data.frame(z = z_r_yield_z, rprime = z_r_yield_r, Label = "polyline"))
    #print(zrprime_data)
    zrprime_trial_closest = zrprime_data[which(zrprime_data$Label == "trial" |
                                               zrprime_data$Label == "closest"),]
    print(zrprime_trial_closest)
    plt = ggplot() + 
          geom_path(data = zrprime_data, 
                    aes(x = z, y = rprime, group=Label, color = Label))+
          geom_point(data = zrprime_data, 
                     aes(x = z, y = rprime, group=Label, color = Label)) +
          geom_line(data = zrprime_trial_closest,
                    aes(x = z, y = rprime),
                    color = "red", linetype = 1) +
          coord_fixed() +
          theme_bw()
    print(plt)
  }
}

#-------------------------------------------------------------------------
# Actually read and plot one timestep
#-------------------------------------------------------------------------
ReadAndPlotT1 <- function() {
  num_points = 50
  consistency_iter = 1
  iteration = 1
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-478.855,1038.89)
  z_r_yield_z = c(577.35,467.086,178.411,-178.411,-467.086,-577.35)
  z_r_yield_r = c(0,208.891,755.774,1361.68,1163.02,0)
  zr_df = ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter)

  iteration = 2
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-487.579,1037.18)
  z_r_yield_z = c(3.53525e-14,-178.411,-339.358,-467.086,-549.093,-577.35)
  z_r_yield_r = c(1093.77,1361.68,1404.99,1163.02,659.442,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-491.522,1036.8)
  z_r_yield_z = c(-288.675,-386.323,-467.086,-527.436,-564.734,-577.35)
  z_r_yield_r = c(1420.84,1356.71,1163.02,851.288,449.844,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-494.55,1037.73)
  z_r_yield_z = c(-433.013,-483.505,-523.912,-553.392,-571.329,-577.35)
  z_r_yield_r = c(1266.05,1098.34,876.614,610.69,313.458,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.413,1037.69)
  z_r_yield_z = c(-433.013,-449.201,-464.539,-479.001,-492.556,-505.181)
  z_r_yield_r = c(1266.05,1221.73,1172.07,1117.23,1057.42,992.845)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.602,1037.47)
  z_r_yield_z = c(-469.097,-476.844,-484.33,-491.551,-498.503,-505.181)
  z_r_yield_r = c(1155.69,1125.96,1094.77,1062.17,1028.18,992.845)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.62,1037.56)
  z_r_yield_z = c(-487.139,-490.896,-494.579,-498.188,-501.722,-505.181)
  z_r_yield_r = c(1082.4,1065.24,1047.69,1029.78,1011.49,992.845)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.625,1037.61)
  z_r_yield_z = c(-496.16,-498.004,-499.828,-501.632,-503.417,-505.181)
  z_r_yield_r = c(1039.93,1030.71,1021.39,1011.97,1002.45,992.845)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.633,1037.58)
  z_r_yield_z = c(-496.16,-497.072,-497.979,-498.881,-499.778,-500.671)
  z_r_yield_r = c(1039.93,1035.4,1030.84,1026.25,1021.64,1017.01)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.632,1037.6)
  z_r_yield_z = c(-496.16,-496.614,-497.066,-497.517,-497.967,-498.416)
  z_r_yield_r = c(1039.93,1037.68,1035.43,1033.17,1030.9,1028.62)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.633,1037.59)
  z_r_yield_z = c(-496.16,-496.386,-496.612,-496.838,-497.063,-497.288)
  z_r_yield_r = c(1039.93,1038.81,1037.69,1036.57,1035.44,1034.32)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 12
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.16,-496.273,-496.386,-496.499,-496.612,-496.724)
  z_r_yield_r = c(1039.93,1039.38,1038.82,1038.26,1037.7,1037.13)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 13
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.442,-496.499,-496.555,-496.611,-496.668,-496.724)
  z_r_yield_r = c(1038.54,1038.26,1037.98,1037.7,1037.41,1037.13)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 14
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.583,-496.611,-496.64,-496.668,-496.696,-496.724)
  z_r_yield_r = c(1037.84,1037.7,1037.56,1037.42,1037.27,1037.13)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 15
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.583,-496.597,-496.611,-496.626,-496.64,-496.654)
  z_r_yield_r = c(1037.84,1037.77,1037.7,1037.63,1037.56,1037.49)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 16
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.618,-496.626,-496.633,-496.64,-496.647,-496.654)
  z_r_yield_r = c(1037.66,1037.63,1037.59,1037.56,1037.52,1037.49)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 17
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.618,-496.622,-496.626,-496.629,-496.633,-496.636)
  z_r_yield_r = c(1037.66,1037.64,1037.63,1037.61,1037.59,1037.57)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 18
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.627,-496.629,-496.631,-496.633,-496.634,-496.636)
  z_r_yield_r = c(1037.62,1037.61,1037.6,1037.59,1037.58,1037.57)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 19
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.632,-496.633,-496.633,-496.634,-496.635,-496.636)
  z_r_yield_r = c(1037.59,1037.59,1037.59,1037.58,1037.58,1037.57)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 20
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.635,-496.635,-496.636,-496.636)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.57)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 21
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.635,-496.635,-496.635)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 22
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.634,-496.634,-496.634)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 23
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.634,-496.634,-496.634)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 24
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.634,-496.634,-496.634)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 25
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.634,-496.634,-496.634)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 26
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.634,-496.634,-496.634)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 27
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, YSLOPE = 0.355)
  z_r_pt = c(-524.814,1043.24)
  z_r_closest = c(-496.634,1037.58)
  z_r_yield_z = c(-496.634,-496.634,-496.634,-496.634,-496.634,-496.634)
  z_r_yield_r = c(1037.58,1037.58,1037.58,1037.58,1037.58,1037.58)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  createGIFIterations(zr_df, consistency_iter)

}

ReadAndPlotTFail <- function() {

  num_points = 50
  consistency_iter = 1

  iteration = 1
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.15079e+06,2.02165e+06)
  z_r_yield_z = c(575.109,-167406,-607187,-1.15079e+06,-1.59057e+06,-1.75855e+06)
  z_r_yield_r = c(2.45257e-09,310133,1.12207e+06,2.02165e+06,1.72669e+06,0)
  zr_df = 
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter)
  iteration = 2
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.15079e+06,2.02165e+06)
  z_r_yield_z = c(-878986,-1.15079e+06,-1.39598e+06,-1.59057e+06,-1.7155e+06,-1.75855e+06)
  z_r_yield_r = c(1.62388e+06,2.02165e+06,2.08595e+06,1.72669e+06,979052,0)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 3
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.15079e+06,2.02165e+06)
  z_r_yield_z = c(-878986,-970925,-1.06186e+06,-1.15079e+06,-1.23674e+06,-1.31877e+06)
  z_r_yield_r = c(1.62388e+06,1.7838e+06,1.91864e+06,2.02165e+06,2.08688e+06,2.10948e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 4
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18969e+06,2.05584e+06)
  z_r_yield_z = c(-1.09888e+06,-1.14468e+06,-1.18969e+06,-1.2338e+06,-1.27687e+06,-1.31877e+06)
  z_r_yield_r = c(1.96539e+06,2.01563e+06,2.05584e+06,2.0853e+06,2.10336e+06,2.10948e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 5
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18723e+06,2.05389e+06)
  z_r_yield_z = c(-1.09888e+06,-1.12123e+06,-1.14342e+06,-1.16542e+06,-1.18723e+06,-1.20882e+06)
  z_r_yield_r = c(1.96539e+06,1.99102e+06,2.01438e+06,2.03536e+06,2.05389e+06,2.06989e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 6
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18699e+06,2.05371e+06)
  z_r_yield_z = c(-1.15385e+06,-1.16495e+06,-1.176e+06,-1.18699e+06,-1.19794e+06,-1.20882e+06)
  z_r_yield_r = c(2.0246e+06,2.03493e+06,2.04464e+06,2.05371e+06,2.06213e+06,2.06989e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 7
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18686e+06,2.0536e+06)
  z_r_yield_z = c(-1.18133e+06,-1.18686e+06,-1.19237e+06,-1.19787e+06,-1.20335e+06,-1.20882e+06)
  z_r_yield_r = c(2.04911e+06,2.0536e+06,2.05792e+06,2.06208e+06,2.06607e+06,2.06989e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 8
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18684e+06,2.05359e+06)
  z_r_yield_z = c(-1.18133e+06,-1.18409e+06,-1.18684e+06,-1.18959e+06,-1.19234e+06,-1.19508e+06)
  z_r_yield_r = c(2.04911e+06,2.05137e+06,2.05359e+06,2.05576e+06,2.05789e+06,2.05999e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 9
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18683e+06,2.05358e+06)
  z_r_yield_z = c(-1.18133e+06,-1.18271e+06,-1.18409e+06,-1.18546e+06,-1.18683e+06,-1.18821e+06)
  z_r_yield_r = c(2.04911e+06,2.05025e+06,2.05137e+06,2.05248e+06,2.05358e+06,2.05467e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 10
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18658e+06,2.05338e+06)
  z_r_yield_z = c(-1.18477e+06,-1.18546e+06,-1.18615e+06,-1.18683e+06,-1.18752e+06,-1.18821e+06)
  z_r_yield_r = c(2.05192e+06,2.05248e+06,2.05303e+06,2.05358e+06,2.05412e+06,2.05467e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 11
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18649e+06,2.0533e+06)
  z_r_yield_z = c(-1.18649e+06,-1.18683e+06,-1.18718e+06,-1.18752e+06,-1.18786e+06,-1.18821e+06)
  z_r_yield_r = c(2.0533e+06,2.05358e+06,2.05385e+06,2.05412e+06,2.0544e+06,2.05467e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  iteration = 12
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.87891e+07
  pbar_w = 5.24774e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.18649e+06,2.0533e+06)
  z_r_yield_z = c(-1.18649e+06,-1.18666e+06,-1.18683e+06,-1.187e+06,-1.18718e+06,-1.18735e+06)
  z_r_yield_r = c(2.0533e+06,2.05344e+06,2.05358e+06,2.05371e+06,2.05385e+06,2.05399e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  createGIFIterations(zr_df, consistency_iter)
}

TestPlotting <- function() {
  K = 5.8393e+07
  G = 1.85581e+07
  X = -1000 
  pbar_w = -0
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.356, PEAKI1 = 1000, STREN = 1.77e+07, T1 = 1e-06, T2 = 0.5, YSLOPE = 0.355)
  z_r_pt = c(-524.814, 1043.24, 0)
  z_r_closest = c(-478.855, 1038.89, 0)
  z_r_yield_z = c(577.35, 467.086, 178.411, -178.411, -467.086, -577.35 )
  z_r_yield_r = c(0, 208.891, 755.774, 1361.68, 1163.02, 0 )
  iteration = 1
  ComputeAndPlotFullYieldSurface(yieldParams, X, pbar_w, K, G, 20, "I1J2")
  ComputeAndPlotFullYieldSurface(yieldParams, X, pbar_w, K, G, 20, "rzprime",
                                 z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r)
}

ReadAndPlotTBeforeFail <- function() {

  consistency_iter = 1
  iteration = 1
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.50236e+06,2.66257e+06)
  z_r_yield_z = c(575.109,-252782,-916080,-1.73596e+06,-2.39926e+06,-2.65262e+06)
  z_r_yield_r = c(2.20564e-09,467758,1.69236e+06,3.04914e+06,2.60428e+06,0)
  zr_df = 
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter)
  iteration = 2
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.42431e+06,2.59305e+06)
  z_r_yield_z = c(-1.32602e+06,-1.73596e+06,-2.10577e+06,-2.39926e+06,-2.58769e+06,-2.65262e+06)
  z_r_yield_r = c(2.44921e+06,3.04914e+06,3.14612e+06,2.60428e+06,1.47665e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.46469e+06,2.6904e+06)
  z_r_yield_z = c(-1.32602e+06,-1.46469e+06,-1.60184e+06,-1.73596e+06,-1.8656e+06,-1.98932e+06)
  z_r_yield_r = c(2.44921e+06,2.6904e+06,2.89378e+06,3.04914e+06,3.14753e+06,3.18162e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45987e+06,2.68258e+06)
  z_r_yield_z = c(-1.32602e+06,-1.39303e+06,-1.45987e+06,-1.52637e+06,-1.59236e+06,-1.65767e+06)
  z_r_yield_r = c(2.44921e+06,2.56965e+06,2.68258e+06,2.78677e+06,2.88106e+06,2.9643e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45881e+06,2.68083e+06)
  z_r_yield_z = c(-1.32602e+06,-1.35927e+06,-1.3925e+06,-1.42568e+06,-1.45881e+06,-1.49184e+06)
  z_r_yield_r = c(2.44921e+06,2.50981e+06,2.56871e+06,2.62577e+06,2.68083e+06,2.73375e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45874e+06,2.68073e+06)
  z_r_yield_z = c(-1.40893e+06,-1.42555e+06,-1.44216e+06,-1.45874e+06,-1.4753e+06,-1.49184e+06)
  z_r_yield_r = c(2.5972e+06,2.62555e+06,2.6534e+06,2.68073e+06,2.70752e+06,2.73375e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45869e+06,2.68065e+06)
  z_r_yield_z = c(-1.45039e+06,-1.45869e+06,-1.46699e+06,-1.47528e+06,-1.48357e+06,-1.49184e+06)
  z_r_yield_r = c(2.66703e+06,2.68065e+06,2.69413e+06,2.70748e+06,2.72069e+06,2.73375e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45787e+06,2.67931e+06)
  z_r_yield_z = c(-1.45039e+06,-1.45454e+06,-1.45868e+06,-1.46283e+06,-1.46697e+06,-1.47112e+06)
  z_r_yield_r = c(2.66703e+06,2.67385e+06,2.68063e+06,2.68739e+06,2.69411e+06,2.70079e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.4574e+06,2.67853e+06)
  z_r_yield_z = c(-1.45039e+06,-1.45246e+06,-1.45454e+06,-1.45661e+06,-1.45868e+06,-1.46075e+06)
  z_r_yield_r = c(2.66703e+06,2.67044e+06,2.67384e+06,2.67724e+06,2.68063e+06,2.68401e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45764e+06,2.67893e+06)
  z_r_yield_z = c(-1.45557e+06,-1.45661e+06,-1.45764e+06,-1.45868e+06,-1.45972e+06,-1.46075e+06)
  z_r_yield_r = c(2.67554e+06,2.67724e+06,2.67894e+06,2.68063e+06,2.68232e+06,2.68401e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45752e+06,2.67873e+06)
  z_r_yield_z = c(-1.45557e+06,-1.45609e+06,-1.45661e+06,-1.45713e+06,-1.45764e+06,-1.45816e+06)
  z_r_yield_r = c(2.67554e+06,2.67639e+06,2.67724e+06,2.67809e+06,2.67894e+06,2.67978e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 12
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45746e+06,2.67863e+06)
  z_r_yield_z = c(-1.45687e+06,-1.45713e+06,-1.45738e+06,-1.45764e+06,-1.4579e+06,-1.45816e+06)
  z_r_yield_r = c(2.67766e+06,2.67809e+06,2.67851e+06,2.67894e+06,2.67936e+06,2.67978e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 13
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45749e+06,2.67868e+06)
  z_r_yield_z = c(-1.45687e+06,-1.457e+06,-1.45713e+06,-1.45725e+06,-1.45738e+06,-1.45751e+06)
  z_r_yield_r = c(2.67766e+06,2.67788e+06,2.67809e+06,2.6783e+06,2.67851e+06,2.67872e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 14
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45747e+06,2.67866e+06)
  z_r_yield_z = c(-1.45719e+06,-1.45725e+06,-1.45732e+06,-1.45738e+06,-1.45745e+06,-1.45751e+06)
  z_r_yield_r = c(2.67819e+06,2.6783e+06,2.67841e+06,2.67851e+06,2.67862e+06,2.67872e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 15
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67867e+06)
  z_r_yield_z = c(-1.45735e+06,-1.45738e+06,-1.45742e+06,-1.45745e+06,-1.45748e+06,-1.45751e+06)
  z_r_yield_r = c(2.67846e+06,2.67851e+06,2.67856e+06,2.67862e+06,2.67867e+06,2.67872e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 16
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45743e+06,-1.45745e+06,-1.45747e+06,-1.45748e+06,-1.4575e+06,-1.45751e+06)
  z_r_yield_r = c(2.67859e+06,2.67862e+06,2.67864e+06,2.67867e+06,2.6787e+06,2.67872e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 17
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45747e+06,-1.45748e+06,-1.45749e+06,-1.4575e+06,-1.45751e+06,-1.45751e+06)
  z_r_yield_r = c(2.67866e+06,2.67867e+06,2.67868e+06,2.6787e+06,2.67871e+06,2.67872e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 18
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45747e+06,-1.45748e+06,-1.45748e+06,-1.45749e+06,-1.45749e+06,-1.45749e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67867e+06,2.67868e+06,2.67868e+06,2.67869e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 19
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45747e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67867e+06,2.67867e+06,2.67867e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 20
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45747e+06,-1.45747e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67867e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 21
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67867e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 22
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 23
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 24
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 25
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 26
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.86817e+08
  pbar_w = 6.07408e+07
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(52172.3,3.60195e+06)
  z_r_closest = c(-1.45748e+06,2.67866e+06)
  z_r_yield_z = c(-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06,-1.45748e+06)
  z_r_yield_r = c(2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06,2.67866e+06)
  zr_df = rbind(zr_df,
  ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                          z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                          iteration, consistency_iter))
  createGIFIterations(zr_df, consistency_iter)
}

ReadAndPlotTFail7 <- function() {

  num_points = 50

  consistency_iter = 1
  iteration = 1
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.07133e+06,1.92937e+06)
  z_r_yield_z = c(575.109,-86255.1,-329548,-681116,-1.07133e+06,-1.4229e+06,-1.66619e+06,-1.75302e+06)
  z_r_yield_r = c(1.19359e-08,160309,609486,1.25856e+06,1.92937e+06,2.0547e+06,1.33517e+06,0)
  zr_df = 
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter)
  iteration = 2
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.25525e+06,2.09005e+06)
  z_r_yield_z = c(-876222,-1.07133e+06,-1.25665e+06,-1.4229e+06,-1.56173e+06,-1.66619e+06,-1.73104e+06,-1.75302e+06)
  z_r_yield_r = c(1.61878e+06,1.92937e+06,2.09127e+06,2.0547e+06,1.79838e+06,1.33517e+06,711392,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.19655e+06,2.0574e+06)
  z_r_yield_z = c(-876222,-941745,-1.0069e+06,-1.07133e+06,-1.13466e+06,-1.19655e+06,-1.25665e+06,-1.31462e+06)
  z_r_yield_r = c(1.61878e+06,1.73488e+06,1.83927e+06,1.92937e+06,2.0028e+06,2.0574e+06,2.09127e+06,2.10285e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.19229e+06,2.05425e+06)
  z_r_yield_z = c(-1.09542e+06,-1.12811e+06,-1.16041e+06,-1.19229e+06,-1.2237e+06,-1.25459e+06,-1.28491e+06,-1.31462e+06)
  z_r_yield_r = c(1.95922e+06,1.99598e+06,2.02774e+06,2.05425e+06,2.07523e+06,2.09046e+06,2.09973e+06,2.10285e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18395e+06,2.04766e+06)
  z_r_yield_z = c(-1.09542e+06,-1.11136e+06,-1.12721e+06,-1.14297e+06,-1.15864e+06,-1.17421e+06,-1.18967e+06,-1.20502e+06)
  z_r_yield_r = c(1.95922e+06,1.97769e+06,1.99503e+06,2.01118e+06,2.02612e+06,2.03983e+06,2.05226e+06,2.06339e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1817e+06,2.046e+06)
  z_r_yield_z = c(-1.15022e+06,-1.15813e+06,-1.16601e+06,-1.17387e+06,-1.1817e+06,-1.1895e+06,-1.19728e+06,-1.20502e+06)
  z_r_yield_r = c(2.01824e+06,2.02566e+06,2.03276e+06,2.03954e+06,2.046e+06,2.05213e+06,2.05793e+06,2.06339e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18156e+06,2.04588e+06)
  z_r_yield_z = c(-1.17762e+06,-1.18156e+06,-1.18548e+06,-1.18941e+06,-1.19332e+06,-1.19723e+06,-1.20113e+06,-1.20502e+06)
  z_r_yield_r = c(2.04267e+06,2.04588e+06,2.04901e+06,2.05206e+06,2.05502e+06,2.05789e+06,2.06068e+06,2.06339e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18154e+06,2.04587e+06)
  z_r_yield_z = c(-1.17762e+06,-1.17958e+06,-1.18154e+06,-1.1835e+06,-1.18546e+06,-1.18741e+06,-1.18937e+06,-1.19132e+06)
  z_r_yield_r = c(2.04267e+06,2.04428e+06,2.04587e+06,2.04744e+06,2.04899e+06,2.05052e+06,2.05203e+06,2.05351e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18252e+06,2.04666e+06)
  z_r_yield_z = c(-1.17762e+06,-1.1786e+06,-1.17958e+06,-1.18056e+06,-1.18154e+06,-1.18252e+06,-1.18349e+06,-1.18447e+06)
  z_r_yield_r = c(2.04267e+06,2.04348e+06,2.04428e+06,2.04508e+06,2.04587e+06,2.04666e+06,2.04744e+06,2.04821e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18243e+06,2.04659e+06)
  z_r_yield_z = c(-1.18105e+06,-1.18153e+06,-1.18202e+06,-1.18251e+06,-1.183e+06,-1.18349e+06,-1.18398e+06,-1.18447e+06)
  z_r_yield_r = c(2.04547e+06,2.04587e+06,2.04626e+06,2.04665e+06,2.04705e+06,2.04744e+06,2.04782e+06,2.04821e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18227e+06,2.04646e+06)
  z_r_yield_z = c(-1.18105e+06,-1.18129e+06,-1.18153e+06,-1.18178e+06,-1.18202e+06,-1.18227e+06,-1.18251e+06,-1.18276e+06)
  z_r_yield_r = c(2.04547e+06,2.04567e+06,2.04587e+06,2.04606e+06,2.04626e+06,2.04646e+06,2.04665e+06,2.04685e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 12
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18227e+06,2.04646e+06)
  z_r_yield_z = c(-1.1819e+06,-1.18202e+06,-1.18215e+06,-1.18227e+06,-1.18239e+06,-1.18251e+06,-1.18264e+06,-1.18276e+06)
  z_r_yield_r = c(2.04616e+06,2.04626e+06,2.04636e+06,2.04646e+06,2.04656e+06,2.04665e+06,2.04675e+06,2.04685e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 13
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.18229e+06,2.04648e+06)
  z_r_yield_z = c(-1.1819e+06,-1.18196e+06,-1.18202e+06,-1.18209e+06,-1.18215e+06,-1.18221e+06,-1.18227e+06,-1.18233e+06)
  z_r_yield_r = c(2.04616e+06,2.04621e+06,2.04626e+06,2.04631e+06,2.04636e+06,2.04641e+06,2.04646e+06,2.04651e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 14
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.18212e+06,-1.18215e+06,-1.18218e+06,-1.18221e+06,-1.18224e+06,-1.18227e+06,-1.1823e+06,-1.18233e+06)
  z_r_yield_r = c(2.04634e+06,2.04636e+06,2.04638e+06,2.04641e+06,2.04643e+06,2.04646e+06,2.04648e+06,2.04651e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 15
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.18222e+06,-1.18224e+06,-1.18225e+06,-1.18227e+06,-1.18228e+06,-1.1823e+06,-1.18231e+06,-1.18233e+06)
  z_r_yield_r = c(2.04642e+06,2.04643e+06,2.04645e+06,2.04646e+06,2.04647e+06,2.04648e+06,2.04649e+06,2.04651e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 16
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.18228e+06,-1.18228e+06,-1.18229e+06,-1.1823e+06,-1.18231e+06,-1.18231e+06,-1.18232e+06,-1.18233e+06)
  z_r_yield_r = c(2.04646e+06,2.04647e+06,2.04648e+06,2.04648e+06,2.04649e+06,2.04649e+06,2.0465e+06,2.04651e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 17
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.18228e+06,-1.18228e+06,-1.18228e+06,-1.18229e+06,-1.18229e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04646e+06,2.04647e+06,2.04647e+06,2.04647e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04649e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 18
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.18229e+06,-1.18229e+06,-1.18229e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04647e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04649e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 19
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04649e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 20
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 21
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 22
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 23
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 24
  K = 3.98447e+07
  G = 1.32816e+07
  X = -1.75982e+07
  pbar_w = 4.85396e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355309, PEAKI1 = 996.118, STREN = 1.76382e+07, YSLOPE = 0.353622)
  z_r_pt = c(65476.8,3.60183e+06)
  z_r_closest = c(-1.1823e+06,2.04648e+06)
  z_r_yield_z = c(-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06,-1.1823e+06)
  z_r_yield_r = c(2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06,2.04648e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  createGIFIterations(zr_df, consistency_iter)
}

ReadAndPlotTFailtimestep <- function() {

  num_points = 50
  consistency_iter = 1
  iteration = 1
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.24553e+06,1.57005e+06)
  z_r_yield_z = c(575.317,-127408,-462472,-876634,-1.2117e+06,-1.33968e+06)
  z_r_yield_r = c(4.73457e-09,383328,1.38689e+06,2.49878e+06,2.13421e+06,0)
  zr_df = 
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter)
  iteration = 2
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.30688e+06,1.21012e+06)
  z_r_yield_z = c(-669553,-876634,-1.06344e+06,-1.2117e+06,-1.30688e+06,-1.33968e+06)
  z_r_yield_r = c(2.00713e+06,2.49878e+06,2.57825e+06,2.13421e+06,1.21012e+06,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 3
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.28175e+06,1.56217e+06)
  z_r_yield_z = c(-1.00462e+06,-1.11796e+06,-1.2117e+06,-1.28175e+06,-1.32504e+06,-1.33968e+06)
  z_r_yield_r = c(2.60734e+06,2.48966e+06,2.13421e+06,1.56217e+06,825492,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 4
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.28763e+06,1.46639e+06)
  z_r_yield_z = c(-1.17215e+06,-1.23076e+06,-1.27766e+06,-1.31187e+06,-1.33269e+06,-1.33968e+06)
  z_r_yield_r = c(2.32329e+06,2.01552e+06,1.60864e+06,1.12066e+06,575215,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 5
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.2876e+06,1.48696e+06)
  z_r_yield_z = c(-1.25592e+06,-1.28566e+06,-1.30911e+06,-1.32604e+06,-1.33626e+06,-1.33968e+06)
  z_r_yield_r = c(1.82193e+06,1.51542e+06,1.17128e+06,797739,404006,0)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 6
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.2913e+06,1.44308e+06)
  z_r_yield_z = c(-1.25592e+06,-1.26538e+06,-1.27431e+06,-1.28269e+06,-1.29052e+06,-1.2978e+06)
  z_r_yield_r = c(1.82193e+06,1.73535e+06,1.64501e+06,1.55108e+06,1.45376e+06,1.35325e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 7
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29322e+06,1.41757e+06)
  z_r_yield_z = c(-1.27686e+06,-1.28137e+06,-1.28572e+06,-1.28991e+06,-1.29394e+06,-1.2978e+06)
  z_r_yield_r = c(1.61745e+06,1.56655e+06,1.51465e+06,1.46177e+06,1.40797e+06,1.35325e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 8
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.2926e+06,1.42606e+06)
  z_r_yield_z = c(-1.28733e+06,-1.28951e+06,-1.29165e+06,-1.29375e+06,-1.29579e+06,-1.2978e+06)
  z_r_yield_r = c(1.49471e+06,1.46693e+06,1.43889e+06,1.41059e+06,1.38204e+06,1.35325e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 9
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29256e+06,1.42667e+06)
  z_r_yield_z = c(-1.29256e+06,-1.29363e+06,-1.29469e+06,-1.29574e+06,-1.29678e+06,-1.2978e+06)
  z_r_yield_r = c(1.42667e+06,1.41211e+06,1.39749e+06,1.38281e+06,1.36806e+06,1.35325e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 10
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29256e+06,1.42667e+06)
  z_r_yield_z = c(-1.29256e+06,-1.29309e+06,-1.29362e+06,-1.29414e+06,-1.29466e+06,-1.29518e+06)
  z_r_yield_r = c(1.42667e+06,1.4195e+06,1.41232e+06,1.40512e+06,1.39791e+06,1.39068e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 11
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29261e+06,1.42597e+06)
  z_r_yield_z = c(-1.29256e+06,-1.29283e+06,-1.29309e+06,-1.29335e+06,-1.29361e+06,-1.29387e+06)
  z_r_yield_r = c(1.42667e+06,1.42311e+06,1.41955e+06,1.41599e+06,1.41242e+06,1.40885e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 12
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29267e+06,1.42523e+06)
  z_r_yield_z = c(-1.29256e+06,-1.29269e+06,-1.29283e+06,-1.29296e+06,-1.29309e+06,-1.29322e+06)
  z_r_yield_r = c(1.42667e+06,1.42489e+06,1.42312e+06,1.42135e+06,1.41957e+06,1.4178e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 13
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29264e+06,1.4256e+06)
  z_r_yield_z = c(-1.29256e+06,-1.29263e+06,-1.29269e+06,-1.29276e+06,-1.29283e+06,-1.29289e+06)
  z_r_yield_r = c(1.42667e+06,1.42578e+06,1.4249e+06,1.42401e+06,1.42313e+06,1.42224e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 14
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29266e+06,1.42541e+06)
  z_r_yield_z = c(-1.29256e+06,-1.2926e+06,-1.29263e+06,-1.29266e+06,-1.29269e+06,-1.29273e+06)
  z_r_yield_r = c(1.42667e+06,1.42622e+06,1.42578e+06,1.42534e+06,1.4249e+06,1.42446e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 15
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.4255e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29266e+06,-1.29268e+06,-1.29269e+06,-1.29271e+06,-1.29273e+06)
  z_r_yield_r = c(1.42556e+06,1.42534e+06,1.42512e+06,1.4249e+06,1.42468e+06,1.42446e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 16
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42546e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29266e+06,-1.29267e+06,-1.29268e+06,-1.29269e+06)
  z_r_yield_r = c(1.42556e+06,1.42545e+06,1.42534e+06,1.42523e+06,1.42512e+06,1.42501e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 17
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29266e+06,-1.29266e+06,-1.29267e+06)
  z_r_yield_r = c(1.42556e+06,1.42551e+06,1.42545e+06,1.4254e+06,1.42534e+06,1.42529e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 18
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29266e+06)
  z_r_yield_r = c(1.42556e+06,1.42553e+06,1.42551e+06,1.42548e+06,1.42545e+06,1.42542e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 19
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29266e+06)
  z_r_yield_r = c(1.42549e+06,1.42548e+06,1.42547e+06,1.42545e+06,1.42544e+06,1.42542e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 20
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42549e+06,1.42549e+06,1.42548e+06,1.42547e+06,1.42547e+06,1.42546e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 21
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42549e+06,1.42549e+06,1.42549e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 22
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 23
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 24
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 25
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 26
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 27
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 28
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 29
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 30
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))
  iteration = 31
  K = 1.00632e+09
  G = 1.27501e+08
  X = -2.4356e+07
  pbar_w = 7.34521e+06
  yieldParams = list(BETA = 1, CR = 0.5, FSLOPE = 0.355373, PEAKI1 = 996.478, STREN = 1.76439e+07, YSLOPE = 0.35375)
  z_r_pt = c(-1.17472e+07,2.19981e+06)
  z_r_closest = c(-1.29265e+06,1.42548e+06)
  z_r_yield_z = c(-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06,-1.29265e+06)
  z_r_yield_r = c(1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06,1.42548e+06)
  zr_df = rbind(zr_df,
    ComputeFullYieldSurface(yieldParams, X, pbar_w, K, G, num_points,
                            z_r_pt, z_r_closest, z_r_yield_z, z_r_yield_r,
                            iteration, consistency_iter))

  createGIFIterations(zr_df, consistency_iter)
}
#-------------------------------------------------------------------------
# Actually read and plot
#-------------------------------------------------------------------------
#ReadAndPlotTBeforeFail()
ReadAndPlotTFail()
#ReadAndPlotTFail7()
#ReadAndPlotTFailtimestep()
#source("ReadAndPlotTFailPrev.R")
#zr_df = ReadAndPlotTFailPrev()
