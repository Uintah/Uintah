/**************************************************************************
 // Class   : MPMMaterialInputPanel.java
 // Purpose : Create a panel that contains widgets to take inputs for
 //           MPM materials
 // Author  : Biswajit Banerjee
 // Date    : 05/04/2006
 // Mods    :
 //**************************************************************************/

import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.text.*;
import javax.swing.event.*;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// MPM material input panel
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
public class MPMMaterialInputPanel extends JPanel 
                                   implements ItemListener {

  // Data and components
  private boolean d_isRigid = false;
  private String d_burnModel = null;
  private String d_constModel = null;

  private JTextField matNameEntry = null;
  private DecimalField densityEntry = null;
  private DecimalField thermalCondEntry = null;
  private DecimalField spHeatEntry = null;
  private DecimalField roomTempEntry = null;
  private DecimalField meltTempEntry = null;

  private JComboBox burnModelComB = null;
  private JComboBox constModelComB = null;
  private JCheckBox isRigidCB = null;

  private JPanel constModelPanel = null;
  private RigidMaterialPanel rigidPanel = null;
  private ElasticPlasticMaterialPanel elasticPlasticPanel = null;
  private ViscoSCRAMMaterialPanel viscoSCRAMPanel = null;

  public MPMMaterialInputPanel(int matIndex) {

    // Initialize
    d_isRigid = false;
    d_constModel = new String("rigid");
    d_burnModel = new String("null");

    // Create a gridbaglayout and constraints
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Add the first panel
    JPanel panel1 = new JPanel(new GridLayout(6,0));

    JLabel matNameLabel = new JLabel("Material Name");
    String matName = new String("Material "+ String.valueOf(matIndex));
    matNameEntry = new JTextField(matName, 20);
    panel1.add(matNameLabel);
    panel1.add(matNameEntry);

    JLabel densityLabel = new JLabel("Density");
    densityEntry = new DecimalField(8900.0, 7);
    panel1.add(densityLabel);
    panel1.add(densityEntry);

    JLabel thermalCondLabel = new JLabel("Thermal Conductivity");
    thermalCondEntry = new DecimalField(390.0, 7);
    panel1.add(thermalCondLabel);
    panel1.add(thermalCondEntry);

    JLabel spHeatLabel = new JLabel("Specific Heat");
    spHeatEntry = new DecimalField(410.0, 7);
    panel1.add(spHeatLabel);
    panel1.add(spHeatEntry);

    JLabel roomTempLabel = new JLabel("Room Temperature");
    roomTempEntry = new DecimalField(298.0, 7);
    panel1.add(roomTempLabel);
    panel1.add(roomTempEntry);

    JLabel meltTempLabel = new JLabel("Melt Temperature");
    meltTempEntry = new DecimalField(1400.0, 7);
    panel1.add(meltTempLabel);
    panel1.add(meltTempEntry);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 0, 1, 1, 5);
    gb.setConstraints(panel1, gbc);
    add(panel1); 

    // Add the second panel
    JPanel panel2 = new JPanel(new GridLayout(6,2));

    JLabel burnModelLabel = new JLabel("Burn Model");
    burnModelComB = new JComboBox();
    burnModelComB.addItem("None");
    burnModelComB.addItem("Simple Burn");
    burnModelComB.addItem("Pressure Burn");
    burnModelComB.addItem("Ignition and Combustion");
    panel2.add(burnModelLabel); panel2.add(burnModelComB); 

    JLabel constModelLabel = new JLabel("Constitutive Model");
    constModelComB = new JComboBox();
    constModelComB.addItem("Rigid");
    constModelComB.addItem("Hypoelastic");
    constModelComB.addItem("Compressible Neo-Hookean");
    constModelComB.addItem("Elastic-Plastic");
    constModelComB.addItem("ViscoSCRAM");
    panel2.add(constModelLabel); panel2.add(constModelComB); 

    isRigidCB = new JCheckBox("Treat Material As Rigid ");
    isRigidCB.setSelected(false);
    isRigidCB.setEnabled(false);
    panel2.add(isRigidCB);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 0, 1, 1, 1, 5);
    gb.setConstraints(panel2, gbc);
    add(panel2); 

    // Add the Third panel for the constituive models
    constModelPanel = new JPanel();
    rigidPanel = new RigidMaterialPanel();
    //hypoElasticPanel = new HypoElasticMaterialPanel();
    //cnhElasticPanel = new CompNeoHookMaterialPanel();
    elasticPlasticPanel = new ElasticPlasticMaterialPanel();
    viscoSCRAMPanel = new ViscoSCRAMMaterialPanel();
    constModelPanel.add(rigidPanel);
    constModelPanel.add(elasticPlasticPanel);
    constModelPanel.add(viscoSCRAMPanel);
    rigidPanel.setVisible(true);
    elasticPlasticPanel.setVisible(false);
    viscoSCRAMPanel.setVisible(false);

    UintahGui.setConstraints(gbc, GridBagConstraints.BOTH, 
                             1.0, 1.0, 1, 0, 
                             1, GridBagConstraints.REMAINDER, 5);
    gb.setConstraints(constModelPanel, gbc);
    add(constModelPanel); 

    // Create and add the listeners
    burnModelComB.addItemListener(this);
    constModelComB.addItemListener(this);

    CheckBoxListener checkBoxListener = new CheckBoxListener();
    isRigidCB.addItemListener(checkBoxListener);
  }

  //-----------------------------------------------------------------------
  // Gets the name of the material
  //-----------------------------------------------------------------------
  public String getName() {
    return matNameEntry.getText();
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Listens for item picked in combo box and takes action as required.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  public void itemStateChanged(ItemEvent e) {
        
    // Get the object that has been selected
    Object source = e.getItemSelectable();

    // Get the item that has been selected
    String item = String.valueOf(e.getItem());

    if (source == burnModelComB) {
      if (item.equals(new String("None"))) {
        d_burnModel = "null";
      } else if (item.equals(new String("Simple Burn"))) {
        d_burnModel = "simple";
      } else if (item.equals(new String("Pressure Burn"))) {
        d_burnModel = "pressure";
      } else if (item.equals(new String("Ignition and Combustion"))) {
        d_burnModel = "IgnitionCombustion";
      }
    } else {
      d_isRigid = false;
      if (item == "Rigid") {
        d_constModel = "rigid";
        rigidPanel.setVisible(true);
        elasticPlasticPanel.setVisible(false);
        viscoSCRAMPanel.setVisible(false);
        d_isRigid = true;
        isRigidCB.setSelected(true);
        isRigidCB.setEnabled(false);
        validate();
      } else if (item == "Hypoelastic") {
        d_constModel = "hypoelastic";
        rigidPanel.setVisible(false);
        elasticPlasticPanel.setVisible(false);
        viscoSCRAMPanel.setVisible(false);
        isRigidCB.setSelected(false);
        isRigidCB.setEnabled(true);
        validate();
      } else if (item == "Compressible Neo-Hookean") {
        d_constModel = "comp_neo_hook";
        rigidPanel.setVisible(false);
        elasticPlasticPanel.setVisible(false);
        viscoSCRAMPanel.setVisible(false);
        isRigidCB.setSelected(false);
        isRigidCB.setEnabled(true);
        validate();
      } else if (item == "Elastic-Plastic") {
        d_constModel = "elastic_plastic";
        rigidPanel.setVisible(false);
        elasticPlasticPanel.setVisible(true);
        viscoSCRAMPanel.setVisible(false);
        isRigidCB.setSelected(false);
        isRigidCB.setEnabled(true);
        validate();
      } else if (item == "ViscoSCRAM") {
        d_constModel = "visco_scram";
        rigidPanel.setVisible(false);
        elasticPlasticPanel.setVisible(false);
        viscoSCRAMPanel.setVisible(true);
        isRigidCB.setSelected(false);
        isRigidCB.setEnabled(true);
        validate();
      }
    }
  }

  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // Class   : CheckBoxListener
  // Purpose : Listens for item seleceted in check box and takes action as
  //           required.
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  private class CheckBoxListener implements ItemListener {
    public void itemStateChanged(ItemEvent e) {
        
      if (e.getStateChange() == ItemEvent.SELECTED) {
        d_isRigid = true;
      } else {
        d_isRigid = false;
      }
    }
  }

  //--------------------------------------------------------------------
  /** Write the contents out in Uintah format */
  //--------------------------------------------------------------------
  public void writeUintah(PrintWriter pw, String tab) {
      
    if (pw == null) return;

    // Write the data
    String tab1 = new String(tab+"  ");
    String tab2 = new String(tab1+"  ");
    pw.println(tab+"<material name = \""+matNameEntry.getText()+"\">");
    pw.println(tab1+"<is_rigid> "+d_isRigid+" </is_rigid>");
    pw.println(tab1+"<density> "+densityEntry.getValue()+
               " </density>");
    pw.println(tab1+"<thermal_conductivity> "+thermalCondEntry.getValue()+
               " </thermal_conductivity>");
    pw.println(tab1+"<specific_heat> "+spHeatEntry.getValue()+
               " </specific_heat>");
    pw.println(tab1+"<room_temp> "+roomTempEntry.getValue()+
               " </room_temp>");
    pw.println(tab1+"<melt_temp> "+meltTempEntry.getValue()+
               " </melt_temp>");
    pw.println(tab);

    pw.println(tab1+"<constitutive_model type=\""+d_constModel+"\">");
    if (d_constModel.equals(new String("rigid"))) {
      rigidPanel.writeUintah(pw, tab2);
    } else if (d_constModel.equals(new String("elastic_plastic"))) {
      elasticPlasticPanel.writeUintah(pw, tab2);
    }
    pw.println(tab1+"</constitutive_model>");
    pw.println(tab+"</material>");
    pw.println(tab);

  }
}
