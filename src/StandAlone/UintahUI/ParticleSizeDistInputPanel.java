//**************************************************************************
// Program : ParticleSizeDistInputPanel.java
// Purpose : Input particle size distribution and plot size distribution.
// Author  : Biswajit Banerjee
// Date    : 05/04/2006
// Mods    :
//**************************************************************************

//************ IMPORTS **************
import java.awt.*;
import javax.swing.*;

//**************************************************************************
// Class   : ParticleSizeDistInputPanel
// Purpose : Inputs particle size distribution and plot size histogram.
//**************************************************************************
public class ParticleSizeDistInputPanel extends JPanel {

  // Data
  private ParticleGeneratePanel d_parentPanel = null;
  private ParticleSize d_partSizeDist = null;

  private InputPartDistPanel inputPanel = null;
  private DisplayPartDistFrame displayFrame = null;

  // Constructor
  public ParticleSizeDistInputPanel(ParticleSize partSizeDist,
                                    ParticleGeneratePanel parentPanel) {

    // Copy the arguments
    d_partSizeDist = partSizeDist;
    d_parentPanel = parentPanel;

    // Create and add the relevant panels
    inputPanel = new InputPartDistPanel(d_partSizeDist, this);
    displayFrame = new DisplayPartDistFrame(d_partSizeDist, this);
    displayFrame.pack();
    displayFrame.setVisible(false);

    // Create a grid bag
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints gbc = new GridBagConstraints();
    setLayout(gb);

    // Set the constraints
    UintahGui.setConstraints(gbc, GridBagConstraints.CENTER,
				    1.0,1.0, 0,0, 1,1,5);
    gb.setConstraints(inputPanel,gbc);
    add(inputPanel);
  }

  public ParticleGeneratePanel getSuper() {
    return d_parentPanel;
  }

  public void refreshDisplayPartDistFrame() {
    displayFrame.refresh();
  }

  public void setVisibleDisplayFrame(boolean visible) {
    System.out.println("part size dist set visible = "+visible);
    displayFrame.setVisible(visible);
  }
}
