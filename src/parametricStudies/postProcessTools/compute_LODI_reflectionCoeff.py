#! /usr/bin/env python3
#______________________________________________________________________
#  compute_LODI_reflectionCoeff.py
#
#  Post-process script for the LODI outflow reflection-coefficient test
#  test_config_files/ICE/Lodi_reflectionCoeff.tst, 
#  inputs/ICE/Lodi_reflectionCoeff.ups).
#
#  The test's initial condition is an off-center, narrow, small-amplitude
#  Gaussian temperature spike at rest in a 1D (in x) domain. At rest, it
#  decomposes into two equal acoustic wave packets running toward each
#  boundary (plus a stationary entropy anomaly that never reaches the
#  probe). The probe is placed, and the spike's location chosen, so that
#  any reflection returning from x- arrives at the probe well after the
#  x+ reflection being measured here -- see the .ups file's own comment
#  block for the timing argument.
#
#  The probe's press_CC/vel_CC history is read from 
#  DataAnalysis:lineExtract module -- 
#
#  Decomposing that history into right-/left-going Riemann invariants
#
#      p_plus  = 0.5*(p - p_inf) + 0.5*rho0*c0*u      (right-going, incident)
#      p_minus = 0.5*(p - p_inf) - 0.5*rho0*c0*u      (left-going, reflected)
#
#  isolates the incident wave (peaks in p_plus near t1) from the x+
#  reflection (peaks in p_minus near t2). This script reports
#
#      |R| = max|p_minus| / max|p_plus|
#
#  the LODI outflow's reflection coefficient for whatever sigma that
#  uda's run used.
#
#  Usage:
#      compute_LODI_reflectionCoeff.py -o <outputFile> -uda <udaName>
#                                       [-dir x|y|z] [-probe <cellIndex>]
#
#  -dir   selects which axis the 1D domain and its LODI x+/y+/z+-style
#         boundary runs along (default: x). A y- or z-direction variant of
#         the .ups (same geometry, permuted axes, following the same
#         convention as this codebase's other _dx/_dy/_dz test families)
#         is expected to place its probe at the same cell-index magnitude
#         along the active axis, with the other two axes pinned to 0 (the
#         thin, symmetric directions), and to name its lineExtract line
#         "probe" (matching LINE_NAME below).
#
#  -probe overrides the probe cell index along the active axis (default:
#         320, matching Lodi_reflectionCoeff.ups's x=1.6 m probe).
#
#  -window <lo> <hi>  overrides the time window (seconds) searched for the
#         reflected peak (default: REFLECTION_WINDOW below, sized for the
#         1D x/y/z tests). 2D/3D variants with different domain sizes or
#         probe placements arrive at different times -- see that .ups
#         file's own timing argument for the values to use here.
#
#  -index <i> <j> <k>  overrides the full probe cell index (ignoring
#         -probe). Needed for 2D/3D variants whose probe isn't on the
#         domain's principal direction  at index 0 along the other
#         two axes -- e.g. the 2D x-y plane test's probe sits at a
#         specific y cell, not y=0. -dir still selects which velocity
#         component feeds the Riemann decomposition.
#
#  References:
#  [1] J.C. Sutherland, C.A. Kennedy, "Improved boundary conditions for
#      viscous, reacting, compressible flows," Journal of Computational
#      Physics, 191(2):502-524, 2003. (doi:10.1016/j.jcp.2003.06.001)
#      -- the LODI formulation this test exercises (CCA/Components/ICE/
#      CustomBCs/LODI2.cc cites this paper's Table 5 and Eq. 8-9 directly).
#
#  [2] T.J. Poinsot, S.K. Lele, "Boundary conditions for direct
#      simulations of compressible viscous flows," Journal of
#      Computational Physics, 101(1):104-129, 1992.
#      (doi:10.1016/0021-9991(92)90046-2) -- origin of the NSCBC outflow
#      relaxation approach and the acoustic-pulse/vortex-convection
#      reflection benchmark this test is modeled on.
#
#  [3] D.H. Rudy, J.C. Strikwerda, "A nonreflecting outflow boundary
#      condition for subsonic Navier-Stokes calculations," Journal of
#      Computational Physics, 36(1):55-70, 1980.
#      (doi:10.1016/0021-9991(80)90174-6) -- source of the sigma*c*(1-M^2)
#      outflow relaxation term and the sigma ~ 0.25-0.28 recommendation.
#
#  [4] C.S. Yoo, H.G. Im, "Characteristic boundary conditions for
#      simulations of compressible reacting flows with multi-dimensional,
#      viscous and reaction effects," Combustion Theory and Modelling,
#      11(2):259-286, 2007. (doi:10.1080/13647830600898995) -- the
#      wave-amplitude-decomposition-at-a-probe methodology this script
#      implements follows this paper's validation approach.
#______________________________________________________________________
import sys
import os
import math

# maps -dir to the axis index (0,1,2) used to index IntVector/vel_CC
AXES = {"x": 0, "y": 1, "z": 2}

LINE_NAME     = "probe"              # must match the .ups's <line name = ...>
LEVEL_INDEX   = 0                    # single level, no AMR
DEFAULT_PROBE = 320                  # cell nearest 1.6 m along the active axis
P_INFINITY    = 101325.0
RHO0          = 1.7899909957225715   # ambient density,  matches the .ups geom_object
GAMMA         = 1.289
CV            = 652.9                # specific_heat in the .ups is cv (p = rho*(gamma-1)*cv*T)
T0            = 300.0

# the x+ reflection arrives later for larger sigma (the relaxation term
# acts like a softer spring); this window comfortably covers sigma in
# [0, 2] while staying well clear of the incident peak (~1.5e-3 s) and
# of any eventual x- contamination (~1.0e-2 s, see the .ups comments)
REFLECTION_WINDOW = (3.5e-3, 5.2e-3)


#______________________________________________________________________
#  parse_args
#______________________________________________________________________
def parse_args(argv):
    out_file  = None
    uda       = None
    direction = "x"
    probe     = DEFAULT_PROBE
    window    = REFLECTION_WINDOW
    index     = None
    i = 0
    while i < len( argv ):
        if argv[i] == "-o" and i + 1 < len( argv ):
            out_file = argv[i + 1]
            i += 2
        elif argv[i] == "-uda" and i + 1 < len( argv ):
            uda = argv[i + 1]
            i += 2
        elif argv[i] == "-dir" and i + 1 < len( argv ):
            direction = argv[i + 1]
            i += 2
        elif argv[i] == "-probe" and i + 1 < len( argv ):
            probe = int( argv[i + 1] )
            i += 2
        elif argv[i] == "-window" and i + 2 < len( argv ):
            window = ( float( argv[i + 1] ), float( argv[i + 2] ) )
            i += 3
        elif argv[i] == "-index" and i + 3 < len( argv ):
            index = ( int( argv[i + 1] ), int( argv[i + 2] ), int( argv[i + 3] ) )
            i += 4
        else:
            i += 1
    return out_file, uda, direction, probe, window, index


#______________________________________________________________________
#  probe_index -- builds the (i,j,k) cell index and the vel_CC component
#  (0,1,2 == x,y,z) for the requested direction
#______________________________________________________________________
def probe_index(direction, probe):
    if direction not in AXES:
        sys.stderr.write( "compute_LODI_reflectionCoeff.py: -dir must be "
                           "x, y, or z (got %s)\n" % direction )
        sys.exit( 1 )

    axis = AXES[direction]
    index = [0, 0, 0]
    index[axis] = probe
    return tuple( index ), axis


#______________________________________________________________________
#  read_probe_file -- parses the lineExtract output at
#
#  uda/<LINE_NAME>/L-<LEVEL_INDEX>/i<i>_j<j>_k<k>. Column order (fixed by
#
#  CCA/Components/OnTheFlyAnalysis/lineExtract.cc's printHeader/doAnalysis):
#    1 X_CC  2 Y_CC  3 Z_CC  4 Timestep  5 Time[s]   6 press_CC_0  7 vel_CC_0.x  8 vel_CC_0.y  9 vel_CC_0.z
#  Returns a list of (t, press, vx, vy, vz) tuples.
#______________________________________________________________________
def read_probe_file(path):
    if not os.path.exists( path ):
        sys.stderr.write( "compute_LODI_reflectionCoeff.py: probe file not "
                           "found: %s\n" % path )
        sys.exit( 1 )

    data = []
    with open( path ) as f:
        for line in f:
            if line.startswith( "#" ) or not line.strip():
                continue
            cols = line.split()
            t     = float( cols[4] )
            press = float( cols[5] )
            vx    = float( cols[6] )
            vy    = float( cols[7] )
            vz    = float( cols[8] )
            data.append( (t, press, vx, vy, vz) )
    return data


#______________________________________________________________________
#  peak_amplitude -- returns the (t, value) sample with the largest |value|
#______________________________________________________________________
def peak_amplitude(series):
    peak = series[0]
    for sample in series:
        if abs( sample[1] ) > abs( peak[1] ):
            peak = sample
    return peak


#______________________________________________________________________
#  main
#______________________________________________________________________
def main():
    out_file, uda, direction, probe, window, index_override = parse_args( sys.argv[1:] )

    if uda is None:
        sys.stderr.write( "compute_LODI_reflectionCoeff.py: -uda is required\n" )
        sys.exit( 1 )

    cell_index, axis = probe_index( direction, probe )

    if index_override is not None:
        cell_index = index_override
    i, j, k = cell_index

    probe_path = "%s/%s/L-%i/i%i_j%i_k%i" % (uda, LINE_NAME, LEVEL_INDEX, i, j, k)
    samples = read_probe_file( probe_path )

    c0 = math.sqrt( GAMMA * CV * (GAMMA - 1.0) * T0 )
    vel_index = axis + 2                # sample = (t, press, vx, vy, vz)

    pplus_series  = []
    pminus_series = []

    for sample in samples:
        t     = sample[0]
        press = sample[1]
        du    = sample[vel_index]
        dp = press - P_INFINITY
        pplus_series.append( (t, 0.5 * dp + 0.5 * RHO0 * c0 * du) )
        pminus_series.append( (t, 0.5 * dp - 0.5 * RHO0 * c0 * du) )

    incident_peak = peak_amplitude( pplus_series )

    in_window = []
    for sample in pminus_series:
        if window[0] <= sample[0] <= window[1]:
            in_window.append( sample )

    if not in_window:
        sys.stderr.write( "compute_LODI_reflectionCoeff.py: no samples in the "
                           "expected reflection window %s -- maxTime too short?\n"
                           % (window,) )
        sys.exit( 1 )
    reflected_peak = peak_amplitude( in_window )

    R = abs( reflected_peak[1] ) / abs( incident_peak[1] )

    if out_file:
        with open( out_file, "w" ) as f:
            f.write( "%g\n" % R )

    print( "direction: %s" % direction )
    print( "probe file: %s" % probe_path )
    print( "incident peak:  t=%.6e  p_plus=%.6f"  % incident_peak )
    print( "reflected peak: t=%.6e  p_minus=%.6f" % reflected_peak )
    print( "|R| = %.6f" % R )


if __name__ == "__main__":
    main()
