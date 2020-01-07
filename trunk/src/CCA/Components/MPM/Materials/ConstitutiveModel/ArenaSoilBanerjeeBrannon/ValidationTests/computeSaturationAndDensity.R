# Set working directory
setwd(".")

#"Moisture content" "0% (water to soil weight ratio)"
MasonSandDry <- function(rho_drysand, alpha = 0.0) {
  #  Inputs
  rho_ref       = 1520      # Dry sand (reference)
  #rho_drysand   = 1640      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  #alpha         = 0.0       # Weight of water to weight of sand

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz
  phi_ref = 1.0 - rho_ref/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  # Compute modulus and strength scale factors
  modulus_fac = exp(-phi/(1 - phi) + phi_ref/(1 - phi_ref))
  strength_fac = exp(4.0*modulus_fac*(modulus_fac - 1.0))

  print(paste("Mason sand dry ", rho_drysand, " kg/m^3: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand, " modulus fac = ", modulus_fac, " strength fac = ", strength_fac))
}

#"Moisture content" "0% (water to soil weight ratio)"
#"Initial density" "1.520 g/cc (dry)"
MasonSandDry1520 <- function() {
  #  Inputs
  rho_drysand   = 1520      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  alpha         = 0.0       # Weight of water to weight of sand

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("Mason sand dry 1520 kg/m^3: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))
}

#"Moisture content" "0% (water to soil weight ratio)"
#"Initial density" "1.77 g/cc (dry)"
MasonSandDry1770 <- function() {
  #  Inputs
  rho_drysand   = 1770      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  alpha         = 0.0       # Weight of water to weight of sand

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("Mason sand dry 1770 kg/m^3: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))
}

#"Moisture content" "18.4% (water to soil weight ratio)"
#"Initial density" "1.52 g/cc (dry)"
MasonSand18ww <- function() {
  #  Inputs
  rho_drysand   = 1520      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  alpha         = 0.184     # Weight of water to weight of sand

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("Mason sand 18%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))
}

#"Moisture content" "10% (water to soil weight ratio)"
#"Initial density" "1.30 g/cc (dry)"
BoulderClay10ww <- function() {
  #  Inputs
  rho_drysand   = 1300      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  alpha         = 0.10     # Weight of water to weight of clay

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("BoulderClay 10%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))

  # Due to swelling the saturation is actually 0.9
  Sw = 0.9*Sw
  
  # Compute actual porosity
  phi = alpha/Sw*rho_drysand/rho_water

  # Compute actual density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("Actual: BoulderClay 10%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))
}

#"Moisture content" "12.8% (water to soil weight ratio)"
#"Initial density" "1.30 g/cc (dry)"
BoulderClay13ww <- function() {
  #  Inputs
  rho_drysand   = 1300      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  alpha         = 0.128     # Weight of water to weight of clay

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("BoulderClay 12.8%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))

  # Due to swelling the saturation is actually 0.9
  Sw = 0.9*Sw
  
  # Compute actual porosity
  phi = alpha/Sw*rho_drysand/rho_water

  # Compute actual density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("Actual: BoulderClay 12.8%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))
}

#"Moisture content" "40.8% (water to soil weight ratio)"
#"Initial density" "1.30 g/cc (dry)"
BoulderClay40ww <- function() {
  #  Inputs
  rho_drysand   = 1300      # Dry sand
  rho_quartz    = 2650      # Quartz
  rho_water     = 1000      # Water
  alpha         = 0.408     # Weight of water to weight of clay

  # Compute porosity
  phi = 1.0 - rho_drysand/rho_quartz

  # Compute saturation
  Sw = alpha/phi*rho_drysand/rho_water

  # Compute density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("BoulderClay 40%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))

  # Due to swelling the saturation is actually 0.9
  Sw = 0.9
  
  # Compute actual porosity
  phi = alpha/Sw*rho_drysand/rho_water

  # Compute actual density
  rho_wetsand = rho_drysand + phi*Sw*rho_water

  print(paste("Actual: BoulderClay 40%: phi = ", phi, "Sw = ", Sw, "rho_wetsand = ", rho_wetsand))
}

#"Moisture content" "10% (water to soil weight ratio)"
#"Initial density" "1620 g/cc (dry)"
SandClay <- function(wf_sand, wf_clay, alpha_wet) {

  # Inputs
  rho_sand_ref = 1520
  rho_sand     = 1700
  rho_clay_ref = 1300
  rho_clay     = 1300
  rho_quartz   = 2650
  rho_water    = 1000

  #wf_sand      = 0.8  # weight fraction
  #wf_clay      = 0.2

  alpha_dry    = 0.0
  #alpha_wet    = 0.10 # Weight of water to weight of mixture

  # Compute dry density and volume fractions
  wf_ratio     = wf_sand/wf_clay
  vf_sand      = wf_ratio*rho_clay/(wf_ratio*rho_clay + rho_sand)
  rho_drymix   = rho_clay + vf_sand*(rho_sand - rho_clay)

  vf_sand_ref      = wf_ratio*rho_clay_ref/(wf_ratio*rho_clay_ref + rho_sand_ref)
  rho_drymix_ref   = rho_clay_ref + vf_sand_ref*(rho_sand_ref - rho_clay_ref)

  # Compute porosity
  phi = 1.0 - rho_drymix/rho_quartz
  phi_ref = 1.0 - rho_drymix_ref/rho_quartz

  # Compute saturation
  Sw = alpha_wet/phi*rho_drymix/rho_water

  # Compute wet density
  rho_wetmix = rho_drymix + phi*Sw*rho_water

  print(paste("Sand = ", wf_sand*100, "%, Clay = ", wf_clay*100, 
              "%, Water = ", alpha_wet*100, "% :"))
  print(paste("rho_dry = ", rho_drymix, "vf = ", vf_sand,
              "phi = ", phi, "Sw = ", Sw, "rho_wet = ", rho_wetmix))
  print(paste("rho_dry_ref = ", rho_drymix_ref, "vf_ref = ", vf_sand_ref,
              "phi_ref = ", phi_ref))
}

#MasonSandDry(1520)
#MasonSandDry(1580)
#MasonSandDry(1640)
#MasonSandDry(1700)
#MasonSandDry(1770)
#MasonSandDry1520()
#MasonSandDry1770()
#MasonSand18ww()
#BoulderClay40ww()
#BoulderClay13ww()
MasonSandDry(1700, 0.10)
#SandClay(0.8, 0.2, 0.1)
#SandClay(0.5, 0.5, 0.1)
