//**************************************************************************
// Class   : VariableSaveInputPanel
// Purpose : A panel that contains widgets to take inputs for
//           the variables to be saved.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import java.util.Vector;

public class VariableSaveInputPanel extends JPanel 
                                    implements ItemListener {

  // Data 
  public Vector d_summedVar = null;
  public Vector d_partVar = null;
  public Vector d_gridVar = null;
  public Vector d_cellVar = null;

  public Vector d_summedVarStr = null;
  public Vector d_partVarStr = null;
  public Vector d_gridVarStr = null;
  public Vector d_cellVarStr = null;

  public Vector d_summedVarState = null;
  public Vector d_partVarState = null;
  public Vector d_gridVarState = null;
  public Vector d_cellVarState = null;

  // Components
  private JCheckBox kineticEnergyCB = null;
  private JCheckBox accStrainEnergyCB = null;
  private JCheckBox strainEnergyCB = null;
  private JCheckBox momentumCB = null;
  private JCheckBox totalMassCB = null;
  private JCheckBox centerOfMassCB = null;

  private JCheckBox p_particleIDCB = null;
  private JCheckBox p_positionCB = null;
  private JCheckBox p_massCB = null;
  private JCheckBox p_volumeCB = null;
  private JCheckBox p_temperatureCB = null;
  private JCheckBox p_stressCB = null;
  private JCheckBox p_deformationGradientCB = null;
  private JCheckBox p_displacementCB = null;
  private JCheckBox p_velocityCB = null;
  private JCheckBox p_externalForceCB = null;
  private JCheckBox p_localizedCB = null;
  private JCheckBox p_damageCB = null;
  private JCheckBox p_porosityCB = null;
  private JCheckBox p_plasticStrainCB = null;
  private JCheckBox p_plasticStrainRateCB = null;
  private JCheckBox p_strainRateCB = null;

  private JCheckBox g_massCB = null;
  private JCheckBox g_volumeCB = null;
  private JCheckBox g_velocityCB = null;
  private JCheckBox g_stressCB = null;
  private JCheckBox g_accelerationCB = null;

  private JCheckBox cc_densityCB = null;
  private JCheckBox cc_temperatureCB = null;
  private JCheckBox cc_velocityCB = null;
  private JCheckBox cc_spVolumeCB = null;
  private JCheckBox cc_volFracCB = null;
  private JCheckBox cc_pressureCB = null;
  private JCheckBox cc_equilPressureCB = null;
  private JCheckBox cc_intEnergyLCB = null;
  private JCheckBox cc_intEnergySourceCB = null;
  private JCheckBox cc_TdotCB = null;
  private JCheckBox cc_momentumLCB = null;
  private JCheckBox cc_momentumSourceCB = null;
  private JCheckBox cc_delPDilatateCB = null;

  public VariableSaveInputPanel() {

    // Initialize vector 
    d_summedVar = new Vector();
    d_partVar = new Vector();
    d_gridVar = new Vector();
    d_cellVar = new Vector();

    d_summedVarStr = new Vector();
    d_partVarStr = new Vector();
    d_gridVarStr = new Vector();
    d_cellVarStr = new Vector();

    d_summedVarState = new Vector();
    d_partVarState = new Vector();
    d_gridVarState = new Vector();
    d_cellVarState = new Vector();

    // Create panels for these check boxes
    JPanel panel0 = new JPanel(new GridLayout(1,0));
    JPanel panel1 = new JPanel(new GridLayout(0,1));
    JPanel panel2 = new JPanel(new GridLayout(0,1));
    JPanel panel3 = new JPanel(new GridLayout(0,1));
    JPanel panel4 = new JPanel(new GridLayout(0,1));

    JLabel label0 = new JLabel("Variables to be saved");
    panel0.add(label0);

    JLabel label1 = new JLabel("Summed Variables");
    panel1.add(label1);

    kineticEnergyCB = new JCheckBox("Kinetic Energy");
    strainEnergyCB = new JCheckBox("Inc. Strain Energy");
    accStrainEnergyCB = new JCheckBox("Total Strain Energy");
    momentumCB = new JCheckBox("Momentum");
    totalMassCB = new JCheckBox("Mass");
    centerOfMassCB = new JCheckBox("Center of Mass");

    d_summedVar.addElement(kineticEnergyCB);
    d_summedVarStr.addElement(new String("KineticEnergy"));
    d_summedVar.addElement(strainEnergyCB);
    d_summedVarStr.addElement(new String("StrainEnergy"));
    d_summedVar.addElement(accStrainEnergyCB);
    d_summedVarStr.addElement(new String("AccStrainEnergy"));
    d_summedVar.addElement(momentumCB);
    d_summedVarStr.addElement(new String("CenterOfMassVelocity"));
    d_summedVar.addElement(totalMassCB);
    d_summedVarStr.addElement(new String("TotalMass"));
    d_summedVar.addElement(centerOfMassCB);
    d_summedVarStr.addElement(new String("CenterOfMassPosition"));

    int numSummedVar = d_summedVar.size();
    for (int ii = 0; ii < numSummedVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_summedVar.elementAt(ii);
      checkBox.setSelected(true);
      checkBox.addItemListener(this);
      d_summedVarState.addElement(new Boolean(true));
      panel1.add(checkBox);
    }

    JLabel label2 = new JLabel("Particle Variables");
    panel2.add(label2);

    p_particleIDCB = new JCheckBox("Particle ID");
    p_positionCB = new JCheckBox("Position");
    p_massCB = new JCheckBox("Mass");
    p_volumeCB = new JCheckBox("Volume");
    p_temperatureCB = new JCheckBox("Temperature");
    p_deformationGradientCB = new JCheckBox("Def. Gradient");
    p_displacementCB = new JCheckBox("Displacement");
    p_velocityCB = new JCheckBox("Velocity");
    p_stressCB = new JCheckBox("Stress");
    p_externalForceCB = new JCheckBox("External Force");
    p_strainRateCB = new JCheckBox("Strain Rate");
    p_localizedCB = new JCheckBox("Failed Particles");
    p_damageCB = new JCheckBox("Damage");
    p_porosityCB = new JCheckBox("Porosity");
    p_plasticStrainCB = new JCheckBox("Plastic Strain");
    p_plasticStrainRateCB = new JCheckBox("Plastic Strain Rate");

    d_partVar.addElement(p_particleIDCB);
    d_partVarStr.addElement(new String("p.particleID"));
    d_partVar.addElement(p_positionCB);
    d_partVarStr.addElement(new String("p.x"));
    d_partVar.addElement(p_massCB);
    d_partVarStr.addElement(new String("p.mass"));
    d_partVar.addElement(p_volumeCB);
    d_partVarStr.addElement(new String("p.volume"));
    d_partVar.addElement(p_temperatureCB);
    d_partVarStr.addElement(new String("p.temperature"));
    d_partVar.addElement(p_deformationGradientCB);
    d_partVarStr.addElement(new String("p.deformationMeasure"));
    d_partVar.addElement(p_displacementCB);
    d_partVarStr.addElement(new String("p.displacement"));
    d_partVar.addElement(p_velocityCB);
    d_partVarStr.addElement(new String("p.velocity"));
    d_partVar.addElement(p_stressCB);
    d_partVarStr.addElement(new String("p.stress"));
    d_partVar.addElement(p_externalForceCB);
    d_partVarStr.addElement(new String("p.externalforce"));
    d_partVar.addElement(p_strainRateCB);
    d_partVarStr.addElement(new String("p.strainRate"));
    d_partVar.addElement(p_localizedCB);
    d_partVarStr.addElement(new String("p.localized"));
    d_partVar.addElement(p_damageCB);
    d_partVarStr.addElement(new String("p.damage"));
    d_partVar.addElement(p_porosityCB);
    d_partVarStr.addElement(new String("p.porosity"));
    d_partVar.addElement(p_plasticStrainCB);
    d_partVarStr.addElement(new String("p.plasticStrain"));
    d_partVar.addElement(p_plasticStrainRateCB);
    d_partVarStr.addElement(new String("p.plasticStrainRate"));

    int numPartVar = d_partVar.size();
    for (int ii = 0; ii < numPartVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_partVar.elementAt(ii);
      checkBox.setSelected(true);
      checkBox.addItemListener(this);
      d_partVarState.addElement(new Boolean(true));
      panel2.add(checkBox);
    }
    p_strainRateCB.setSelected(false);
    p_localizedCB.setSelected(false);
    p_damageCB.setSelected(false);
    p_porosityCB.setSelected(false);
    p_plasticStrainCB.setSelected(false);
    p_plasticStrainRateCB.setSelected(false);

    JLabel label3 = new JLabel("Grid Variables");;
    panel3.add(label3);

    g_massCB = new JCheckBox("Mass");
    g_volumeCB = new JCheckBox("Volume");
    g_velocityCB = new JCheckBox("Velocity");
    g_stressCB = new JCheckBox("Stress");
    g_accelerationCB = new JCheckBox("Acceleration");

    d_gridVar.addElement(g_massCB);
    d_gridVarStr.addElement(new String("g.mass"));
    d_gridVar.addElement(g_volumeCB);
    d_gridVarStr.addElement(new String("g.volume"));
    d_gridVar.addElement(g_velocityCB);
    d_gridVarStr.addElement(new String("g.velocity"));
    d_gridVar.addElement(g_stressCB);
    d_gridVarStr.addElement(new String("g.stressFS"));
    d_gridVar.addElement(g_accelerationCB);
    d_gridVarStr.addElement(new String("g.acceleration"));

    int numGridVar = d_gridVar.size();
    for (int ii = 0; ii < numGridVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_gridVar.elementAt(ii);
      checkBox.setSelected(true);
      checkBox.addItemListener(this);
      d_gridVarState.addElement(new Boolean(true));
      panel3.add(checkBox);
    }

    JLabel label4 = new JLabel("Cell-Centered Variables");
    panel4.add(label4);

    cc_densityCB = new JCheckBox("Density");
    cc_temperatureCB = new JCheckBox("Temperature");
    cc_velocityCB = new JCheckBox("Velocity");
    cc_spVolumeCB = new JCheckBox("Specific Volume");
    cc_volFracCB = new JCheckBox("Volume Fraction");
    cc_pressureCB = new JCheckBox("Pressure");
    cc_equilPressureCB = new JCheckBox("Equilibriation Pressure");
    cc_intEnergyLCB = new JCheckBox("Internal Energy");
    cc_intEnergySourceCB = new JCheckBox("Internal Energy Source");
    cc_TdotCB = new JCheckBox("Temperature Rate");
    cc_momentumLCB = new JCheckBox("Momentum");
    cc_momentumSourceCB = new JCheckBox("Momentum Source");
    cc_delPDilatateCB = new JCheckBox("delP Dilatation");

    d_cellVar.addElement(cc_densityCB);
    d_cellVarStr.addElement(new String("rho_CC"));
    d_cellVar.addElement(cc_temperatureCB);
    d_cellVarStr.addElement(new String("temp_CC"));
    d_cellVar.addElement(cc_velocityCB);
    d_cellVarStr.addElement(new String("vel_CC"));
    d_cellVar.addElement(cc_spVolumeCB);
    d_cellVarStr.addElement(new String("sp_vol_CC"));
    d_cellVar.addElement(cc_volFracCB);
    d_cellVarStr.addElement(new String("vol_frac_CC"));
    d_cellVar.addElement(cc_pressureCB);
    d_cellVarStr.addElement(new String("press_CC"));
    d_cellVar.addElement(cc_equilPressureCB);
    d_cellVarStr.addElement(new String("press_equil_CC"));
    d_cellVar.addElement(cc_intEnergyLCB);
    d_cellVarStr.addElement(new String("int_eng_L_CC"));
    d_cellVar.addElement(cc_intEnergySourceCB);
    d_cellVarStr.addElement(new String("intE_source_CC"));
    d_cellVar.addElement(cc_TdotCB);
    d_cellVarStr.addElement(new String("Tdot"));
    d_cellVar.addElement(cc_momentumLCB);
    d_cellVarStr.addElement(new String("mom_L_CC"));
    d_cellVar.addElement(cc_momentumSourceCB);
    d_cellVarStr.addElement(new String("mom_source_CC"));
    d_cellVar.addElement(cc_delPDilatateCB);
    d_cellVarStr.addElement(new String("delP_Dilatate"));

    int numCellVar = d_cellVar.size();
    for (int ii = 0; ii < numCellVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_cellVar.elementAt(ii);
      checkBox.setSelected(true);
      checkBox.addItemListener(this);
      d_cellVarState.addElement(new Boolean(true));
      panel4.add(checkBox);
    }

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Grid bag layout
    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 0, 
                             GridBagConstraints.REMAINDER,1, 5);
    gb.setConstraints(panel0, gbc);
    add(panel0);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 2, 1, 1, 5);
    gb.setConstraints(panel2, gbc);
    add(panel2);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 1, 1, 1, 1, 5);
    gb.setConstraints(panel3, gbc);
    add(panel3);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 1, 2, 1, 1, 5);
    gb.setConstraints(panel4, gbc);
    add(panel4);
  }

  //-----------------------------------------------------------------------
  // Refresh
  //-----------------------------------------------------------------------
  public void refresh() {
  }

  //--------------------------------------------------------------------
  /** Listen for changed item state */
  //--------------------------------------------------------------------
  public void itemStateChanged(ItemEvent e) {

    // Find the check box which has changed
    Object source = e.getItemSelectable();

    // Find which button was selected and toggle the state
    int numSummedVar = d_summedVar.size();
    for (int ii = 0; ii < numSummedVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_summedVar.elementAt(ii);
      if (source == checkBox) {
        if (e.getStateChange() == ItemEvent.DESELECTED) {
          d_summedVarState.setElementAt(new Boolean(false), ii);
        } else {
          d_summedVarState.setElementAt(new Boolean(true), ii);
        }
        return;
      }
    }

    int numPartVar = d_partVar.size();
    for (int ii = 0; ii < numPartVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_partVar.elementAt(ii);
      if (source == checkBox) {
        if (e.getStateChange() == ItemEvent.DESELECTED) {
          d_partVarState.setElementAt(new Boolean(false), ii);
        } else {
          d_partVarState.setElementAt(new Boolean(true), ii);
        }
        return;
      }
    }

    int numGridVar = d_gridVar.size();
    for (int ii = 0; ii < numGridVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_gridVar.elementAt(ii);
      if (source == checkBox) {
        if (e.getStateChange() == ItemEvent.DESELECTED) {
          d_gridVarState.setElementAt(new Boolean(false), ii);
        } else {
          d_gridVarState.setElementAt(new Boolean(true), ii);
        }
        return;
      }
    }

    int numCellVar = d_cellVar.size();
    for (int ii = 0; ii < numCellVar; ++ii) {
      JCheckBox checkBox = (JCheckBox) d_cellVar.elementAt(ii);
      if (source == checkBox) {
        if (e.getStateChange() == ItemEvent.DESELECTED) {
          d_cellVarState.setElementAt(new Boolean(false), ii);
        } else {
          d_cellVarState.setElementAt(new Boolean(true), ii);
        }
        return;
      }
    }
  }

  //--------------------------------------------------------------------
  /** Write the contents out in Uintah format */
  //--------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
      
    if (pw == null) return;

    String tab1 = new String(tab+"  ");

    // Write the data
    int numSummedVar = d_summedVar.size();
    for (int ii = 0; ii < numSummedVar; ++ii) {
      boolean state = 
        ((Boolean) d_summedVarState.elementAt(ii)).booleanValue();
      if (state) {
        String label = (String) d_summedVarStr.elementAt(ii);
        pw.println(tab1+"<save label=\""+label+"\"/>");
      }
    }
           
    int numPartVar = d_partVar.size();
    for (int ii = 0; ii < numPartVar; ++ii) {
      boolean state = 
        ((Boolean) d_partVarState.elementAt(ii)).booleanValue();
      if (state) {
        String label = (String) d_partVarStr.elementAt(ii);
        pw.println(tab1+"<save label=\""+label+"\"/>");
      }
    }
           
    int numGridVar = d_gridVar.size();
    for (int ii = 0; ii < numGridVar; ++ii) {
      boolean state = 
        ((Boolean) d_gridVarState.elementAt(ii)).booleanValue();
      if (state) {
        String label = (String) d_gridVarStr.elementAt(ii);
        pw.println(tab1+"<save label=\""+label+"\"/>");
      }
    }
           
    int numCellVar = d_cellVar.size();
    for (int ii = 0; ii < numCellVar; ++ii) {
      boolean state = 
        ((Boolean) d_cellVarState.elementAt(ii)).booleanValue();
      if (state) {
        String label = (String) d_cellVarStr.elementAt(ii);
        pw.println(tab1+"<save label=\""+label+"\"/>");
      }
    }

    pw.println(tab+"</DataArchiver>");
    pw.println(tab);
  }
}
