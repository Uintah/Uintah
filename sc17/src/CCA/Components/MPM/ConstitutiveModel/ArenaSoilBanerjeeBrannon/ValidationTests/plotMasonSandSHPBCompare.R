################################################################################
# Name:    plotMasonSandSHPBCompare
#
# Purpose: This routine performs the following actions:
#          1) Reads in the data files creaed by "PlotMasonSandSHPBComapre.py"
#          2) Plots the data
#
# Date:    June, 2016
#
# Running: Run using "Rscript plotMasonSandCompare.R" from a Unix shell
#          or source "plotMasonSandCompare.R" from the R command prompt
#
################################################################################
#
# Clean up
# 
rm(list = ls())
for (i in dev.list()) {dev.off(i)}

# Set up environment and libraries
setwd(".")
Sys.setenv(JAVA_HOME="/usr/lib/jvm/java-7-openjdk-amd64/jre")

# Install ggplot2
if (!require(ggplot2)) {
  install.packages("ggplot2") 
  library(ggplot2)
}

# Install data.table
if (!require(data.table)) {
  install.packages("data.table") 
  library(data.table)
}

# Install the reshape library to melt the data into one long data set
if (!require(reshape2)) {
  install.packages("reshape2")
  library(reshape2)
}

# Install grid extra
if (!require(gridExtra)) {
  install.packages("gridExtra")
  library("gridExtra")
}

# Install MASS for writing the data.frame
if (!require(MASS)) {
  install.packages("MASS")
  library(MASS)
}

# Install pracma (for interp1)
if (!require(pracma)) {
  install.packages("pracma")
  library("pracma")
}

# Install latex2exp
if (!require(latex2exp)) {
  install.packages("latex2exp")
  library("latex2exp")
}

data_files = c('MasonSandSHPB_rt_sim_expt.dat',
               'MasonSandSHPB_zt_sim_expt.dat',
               'MasonSandSHPB_qt_sim_expt.dat',
               'MasonSandSHPB_pt_sim_expt.dat',
               'MasonSandSHPB_Srt_sim_expt.dat',
               'MasonSandSHPB_Sat_sim_expt.dat',
               'MasonSandSHPB_SaEa_sim_expt.dat')
xlabels = c('$t$ ($\\mu$s)', '$t$ ($\\mu$s)', '$t$ ($\\mu$s)', '$t$ ($\\mu$s)', '$\\t$ ($\\mu$s)', 
            '$\\t$ ($\\mu$s)', '$\\epsilon_a$')
ylabels = c('$r$ (MPa)', '$z$ (MPa)', '$q$ (MPa)', '$p$ (MPa)', '$\\sigma_r$ (MPa)', 
            '$\\sigma_a$ (MPa)', '$\\sigma_a$ (MPa)')

data_all = data.frame(data_file = data_files, xlabel = xlabels, ylabel = ylabels)
by(data_all, 1:nrow(data_all), 
  function(data) {
    print(data$data_file)
    dt = read.csv(as.character(data$data_file), sep = " ", header = TRUE,
                  strip.white = TRUE, as.is = TRUE)

    field_names = names(dt)

    names(dt) <- c("Strain", "Stress", "Label", "Density", "Type")

    # Plot the data
    dev.new()
    plot1 <- ggplot() +
             geom_path(data=dt[dt$Type == "sim",],
                       aes(x=Strain, y=Stress, group = Density, color = Density, linetype = Type),
                       size = 1) +
             geom_path(data=dt[dt$Type == "expt",],
                       aes(x=Strain, y=Stress, group = Density, color = Density, linetype = Type), 
                       #linetype = 2,
                       size = 1) +
             #scale_color_brewer(palette = "Accent") +
             #scale_linetype_discrete(labels = c("Sim", "Expt"),
             #                        guide = guide_legend(order = 1)) + 
             xlab(TeX(data["xlabel"])) + ylab(TeX(data["ylabel"])) + 
             theme_bw() +
             theme(legend.position = c(0,1), legend.justification = c(0,1),
                   plot.title = element_text(size = 10),
                   axis.title.x = element_text(size=16),
                   axis.title.y = element_text(size=16),
                   axis.text.x = element_text(size=14),
                   axis.text.y = element_text(size=14),
                   legend.text = element_text(size=12))
    print(plot1)
    pdf_file = paste0(unlist(strsplit(as.character(data$data_file), "[.]"))[1], ".pdf")
    dev.copy(pdf, pdf_file)
    dev.off()
  })

