require("ggplot2")
require("animation")
require("latex2exp")
require("pracma")

#------------------------------------------------------
# Extract a single timestep
#------------------------------------------------------
extractTimestep <- function(exptData, simData, yieldSurfData, time_step) {

  print(paste("time_step", time_step))
  exptData_subset = exptData[1:time_step,]
  simData_subset = simData[1:time_step,]
  npts_ys = nrow(yieldSurfData)/nrow(simData)
  ys_start = npts_ys*(time_step - 1) + 1
  ys_end = npts_ys*time_step
  ysData_subset = yieldSurfData[ys_start:ys_end,]
  ysData_start = ysData_subset[1:npts_ys/2,]
  ysData_end = ysData_subset[(npts_ys/2+1): npts_ys,]
  ysData_end = ysData_end[with(ysData_end, order(P)),]
  ysData_subset_upd = rbind(ysData_start, ysData_end)

  return(list(expt = exptData_subset, sim = simData_subset, yield = ysData_subset_upd))
}

#------------------------------------------------------
# Plot single iteration
#------------------------------------------------------
plotYieldSurface <- function(yield_list, ys_initial, ys_final) {

  exptData = yield_list$expt
  simData = yield_list$sim
  ysData = yield_list$yield
  ys_initial$Label = "Initial yield surface"
  ys_final$Label = "Final yield surface"
  
  df = rbind(exptData, simData, ysData)
  df_if = rbind(ys_initial, ys_final)
  
  plt = ggplot(data = df) + 
        geom_path(data = df,
                  aes(x = P, y = Q, group=Label, color = Label), size=1)+
        geom_path(data = df_if,
                  aes(x = P, y = Q, group=Label, color = Label), size=1)+
        xlab(TeX(paste("$p = I_1/3$", "MPa"))) +
        ylab(TeX(paste("$q = \\sqrt{3J_2}$", "MPa"))) +
        scale_x_continuous(limits = c(0, 80)) + 
        scale_y_continuous(limits = c(-80, 80)) + 
        #coord_fixed() +
        theme_bw() +
        theme(legend.justification=c(0,0), legend.position=c(0,0),
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
animateYieldSurf <- function(exptData, simData, yieldSurfData, ys_initial, ys_final) {
  num_steps = nrow(exptData)
  lapply(seq(1, num_steps, 1), 
         function(time_step) {
           print(paste("time_step = ", time_step))
           plotYieldSurface(extractTimestep(exptData, simData, yieldSurfData, time_step), 
                            ys_initial, ys_final)
         })
}

#------------------------------------------------------
# Function to create animated gif
#------------------------------------------------------
createGIF <- function(exptData, simData, yieldSurfData, ys_initial, ys_final) {

  outputGIFFile = paste0("yield_surf_damage", ".gif")
  print(outputGIFFile)

  ani.options(ani.height = 600, ani.width = 600)

  # Save as animation
  saveGIF(animateYieldSurf(exptData, simData, yieldSurfData, ys_initial, ys_final), 
          interval=0.2, 
          movie.name=outputGIFFile)
}


#-------------------------------------------------------------------------
# Process the yield surface data
#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# Actually read and plot
#-------------------------------------------------------------------------
# Read the file
yieldSurfDataDir = "MasonSandUniaxialStrainSHPBwDamage_26"
yieldSurfDataFile = "MasonSandUniaxialStrainSHPBwDamage_26_vs_expt_pq.dat"
fileName = paste0(getwd(), "/", yieldSurfDataDir, "/", yieldSurfDataFile)
data = read.csv(fileName, header = FALSE, sep = ",")

# Separate out yield surface, sim, and expt data sets
yieldSurfData = data[which(data$V4 == "YieldSurface"),]
simData = data[which(data$V4 == "Simulation"),]
exptData = data[which(data$V4 == "Experiment"),]

# Interpolate experimental data to simulation data
exptData_t = simData$V1[2:(nrow(simData)-1)]
exptData_p = interp1(exptData$V1, exptData$V2, simData$V1[2:(nrow(simData)-1)], method="linear")
exptData_q = interp1(exptData$V1, exptData$V3, simData$V1[2:(nrow(simData)-1)], method="linear")

# Remove the first and last element of simData
simData_t = simData$V1[2:(nrow(simData)-1)]
simData_p = simData$V2[2:(nrow(simData)-1)]
simData_q = simData$V3[2:(nrow(simData)-1)]

# Find the size of yieldSurfData and remove first and last set
nrow_ys = nrow(yieldSurfData)
nrow_sim = nrow(simData)
npts_ys = nrow_ys/nrow_sim
ysData_t = yieldSurfData$V1[(npts_ys+1):(nrow(yieldSurfData)-npts_ys)]
ysData_p = yieldSurfData$V2[(npts_ys+1):(nrow(yieldSurfData)-npts_ys)]
ysData_q = yieldSurfData$V3[(npts_ys+1):(nrow(yieldSurfData)-npts_ys)]

# Get the initial yield surface points
ys_initial = yieldSurfData[1:npts_ys,]
ys_final   = yieldSurfData[(nrow(yieldSurfData)-npts_ys+1):nrow(yieldSurfData),]
names(ys_initial) = c("Time", "P", "Q", "Label")
names(ys_final) = c("Time", "P", "Q", "Label")
ys_initial_start = ys_initial[1:(npts_ys/2),]
ys_initial_end = ys_initial[((npts_ys/2)+1): npts_ys,]
ys_initial_end = ys_initial_end[with(ys_initial_end,
                                     order(P)),]
ys_initial_upd = rbind(ys_initial_start, ys_initial_end)
ys_final_start = ys_final[1:(npts_ys/2),]
ys_final_end = ys_final[((npts_ys/2)+1): npts_ys,]
ys_final_end = ys_final_end[with(ys_final_end,
                                     order(P)),]
ys_final_upd = rbind(ys_final_start, ys_final_end)

# Create updated data frames
exptData_upd = data.frame(Time = exptData_t, P = exptData_p, Q = exptData_q,
                          Label = "Experiment")
simData_upd = data.frame(Time = simData_t, P = simData_p, Q = simData_q,
                          Label = "Simulation")
ysData_upd = data.frame(Time = ysData_t, P = ysData_p, Q = ysData_q,
                          Label = "YieldSurface")

# Create animated GIF
createGIF(exptData_upd, simData_upd, ysData_upd, ys_initial_upd, ys_final_upd)
