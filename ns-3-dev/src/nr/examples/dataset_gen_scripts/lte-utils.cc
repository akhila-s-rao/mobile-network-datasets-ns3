/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 *   Copyright (c) 2020 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
 *
 *   This program is free software; you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License version 2 as
 *   published by the Free Software Foundation;
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program; if not, write to the Free Software
 *   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
#include "lte-utils.h"
#include "nr-utils.h"
#include "cellular-network.h"

namespace ns3 {

void
LteUtils::SetLteSimulatorParameters (const Parameters &params, 
                                           const double sector0AngleRad,
                                           NodeContainer enbSector1Container,
                                           NodeContainer enbSector2Container,
                                           NodeContainer enbSector3Container,
                                           NodeContainer ueNodesContainer,
                                           //NodeContainer ueSector1Container,
                                           //NodeContainer ueSector2Container,
                                           //NodeContainer ueSector3Container,
                                           Ptr<PointToPointEpcHelper> &epcHelper,
                                           Ptr<LteHelper> &lteHelper,
                                           NetDeviceContainer &enbSector1NetDev,
                                           NetDeviceContainer &enbSector2NetDev,
                                           NetDeviceContainer &enbSector3NetDev)
                                           //NetDeviceContainer &ueNetDevsContainer)
                                           //NetDeviceContainer &ueSector1NetDev,
                                           //NetDeviceContainer &ueSector2NetDev,
                                           //NetDeviceContainer &ueSector3NetDev)
{
  NS_UNUSED (params.downtiltAngle);

  /*
   *  An example of how the spectrum is being used, for 20 MHz bandwidth..
   *
   *                              centralEarfcnFrequencyBand = 350
   *                                     |
   *         200 RB                200 RB                200 RB
   * |---------------------|---------------------|---------------------|
   *
   *     100RB      100RB      100RB      100RB      100RB      100RB
   * |----------|----------|----------|----------|----------|----------|
   *      DL         UL         DL         UL         DL         UL
   *
   * |-----|----|-----|----|-----|----|-----|----|-----|----|-----|----|
   *     fc_dl      fc_ul      fc_dl      fc_ul      fc_dl      fc_ul
   *
   * For comparison, the 5GLENA FDD NON_OVERLAPPING case is
   *
   * Scenario 0:  sectors NON_OVERLAPPING in frequency
   *
   * FDD scenario 0:
   *
   * |--------Band0--------|--------Band1--------|--------Band2--------|
   * |---------CC0---------|---------CC1---------|---------CC2---------|
   * |---BWP0---|---BWP1---|---BWP2---|---BWP3---|---BWP4---|---BWP5---|
   *
   *   Sector i will go in Bandi
   *   DL in the first BWP, UL in the second BWP
   *
   */

  uint32_t bandwidthBandDlRB;
  uint32_t bandwidthBandUlRB;

  if (params.bandwidthMHz == 20)
    {
      bandwidthBandDlRB = 100;
      bandwidthBandUlRB = 100;
    }
  else if (params.bandwidthMHz == 15)
    {
      bandwidthBandDlRB = 75;
      bandwidthBandUlRB = 75;
    }
  else if (params.bandwidthMHz == 10)
    {
      bandwidthBandDlRB = 50;
      bandwidthBandUlRB = 50;
    }
  else if (params.bandwidthMHz == 5)
    {
      bandwidthBandDlRB = 25;
      bandwidthBandUlRB = 25;
    }
  else
    {
      NS_ABORT_MSG ("The configured bandwidth in MHz not supported:" << params.bandwidthMHz);
    }

  uint32_t centralFrequencyBand0Dl;
  uint32_t centralFrequencyBand0Ul;
  uint32_t centralFrequencyBand1Dl;
  uint32_t centralFrequencyBand1Ul;
  uint32_t centralFrequencyBand2Dl;
  uint32_t centralFrequencyBand2Ul;

  if (params.freqScenario == 0)  // NON_OVERLAPPING
    {
      centralFrequencyBand0Dl = 100; // 2120 MHz
      centralFrequencyBand0Ul = 200;
      centralFrequencyBand1Dl = 300;
      centralFrequencyBand1Ul = 400;
      centralFrequencyBand2Dl = 500;
      centralFrequencyBand2Ul = 600;
    }
  else
    {
      // OVERLAPPING
      centralFrequencyBand0Dl = 100;   // 2120 MHz
      centralFrequencyBand0Ul = 18100; // 1930 MHz
      centralFrequencyBand1Dl = 100;
      centralFrequencyBand1Ul = 18100;
      centralFrequencyBand2Dl = 100;
      centralFrequencyBand2Ul = 18100;
    }

  double txPower;
    //akhila
    //WARNING
  double ueTxPower = 10; //23
  std::string pathlossModel;
  if (params.scenario == "UMa")
    {
      txPower = 43;
      pathlossModel = "ns3::ThreeGppUmaPropagationLossModel";
    }
  else if (params.scenario == "UMi")
    {
      txPower = 30;// used to be 30
      pathlossModel = "ns3::ThreeGppUmiStreetCanyonPropagationLossModel";
      //WARNING
      //pathlossModel = "ns3::FriisPropagationLossModel";
      //Config::SetDefault ("ns3::RangePropagationLossModel::MaxRange", DoubleValue(200));
      //pathlossModel = "ns3::RangePropagationLossModel";
    }
  else if (params.scenario == "RMa")
    {
      txPower = 43;
      pathlossModel = "ns3::ThreeGppRmaPropagationLossModel";
    }
  else
    {
      NS_FATAL_ERROR ("Selected scenario " << params.scenario << " not valid. Valid values: UMa, UMi, RMa");
    }

  lteHelper = CreateObject<LteHelper> ();
  lteHelper->SetEpcHelper (epcHelper);

  // ALL SECTORS AND BANDS configuration
  //temp
  //Config::SetDefault ("ns3::FfMacScheduler::UlCqiFilter", EnumValue (FfMacScheduler::PUSCH_UL_CQI));
  //Config::SetDefault ("ns3::LteSpectrumPhy::CtrlErrorModelEnabled", BooleanValue (false));
   
  Config::SetDefault ("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue (params.rlcUmTxBuffSize)); //default is 10*1024 = 10KB 
  Config::SetDefault ("ns3::LteEnbPhy::TxPower", DoubleValue (txPower));
  Config::SetDefault ("ns3::LteUePhy::TxPower", DoubleValue (ueTxPower));
  //Config::SetDefault ("ns3::LteUePhy::NoiseFigure", DoubleValue (9.0));
  //Config::SetDefault ("ns3::LteUePhy::EnableRlfDetection", BooleanValue (false));
  //Config::SetDefault ("ns3::LteAmc::AmcModel", EnumValue (LteAmc::PiroEW2010));
 
 //WARNING: set this to something   
 lteHelper->SetAttribute ("PathlossModel", StringValue (pathlossModel)); // for each band the same pathloss model
  
  //WARNING: should be enabled
  lteHelper->SetPathlossModelAttribute ("ShadowingEnabled", BooleanValue (true));
  
  if (params.handoverAlgo == "A3Rsrp") 
  {
      lteHelper->SetHandoverAlgorithmType ("ns3::A3RsrpHandoverAlgorithm");
      lteHelper->SetHandoverAlgorithmAttribute ("Hysteresis", 
                                                DoubleValue (3.0)); // default value 3
      lteHelper->SetHandoverAlgorithmAttribute ("TimeToTrigger",
                                              TimeValue (MilliSeconds (256))); // default value 256 
  }
    else if (params.handoverAlgo == "A2A4Rsrq")
    {
        lteHelper->SetHandoverAlgorithmType ("ns3::A2A4RsrqHandoverAlgorithm");
        lteHelper->SetHandoverAlgorithmAttribute ("ServingCellThreshold",
                                            UintegerValue (30));//30
        lteHelper->SetHandoverAlgorithmAttribute ("NeighbourCellOffset",
                                            UintegerValue (1));// 1
    }

  

  if (params.scheduler == "PF")
    {
      lteHelper->SetSchedulerType ("ns3::PfFfMacScheduler");
    }
  else if (params.scheduler == "RR")
    {
      lteHelper->SetSchedulerType ("ns3::RrFfMacScheduler");
    }

  Config::SetDefault ("ns3::LteUePhy::EnableUplinkPowerControl", BooleanValue (params.enableUlPc));

  lteHelper->SetEnbAntennaModelType ("ns3::CosineAntennaModel");
  lteHelper->SetEnbAntennaModelAttribute ("HorizontalBeamwidth", DoubleValue (60)); // used to be 130
  lteHelper->SetEnbAntennaModelAttribute ("MaxGain", DoubleValue (3));// used to be 0
  lteHelper->SetEnbDeviceAttribute ("DlBandwidth", UintegerValue (bandwidthBandDlRB));
  lteHelper->SetEnbDeviceAttribute ("UlBandwidth", UintegerValue (bandwidthBandUlRB));

  double orientationDegrees = 0;   
  //SECTOR 1 eNB configuration
  orientationDegrees = sector0AngleRad * 180.0 / M_PI;
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (orientationDegrees));
  std::cout << "Sector 1 antenna orientation wrt x axis: " << orientationDegrees << " degrees" << std::endl;
  
  lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (centralFrequencyBand0Dl));
  lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (centralFrequencyBand0Ul));
  enbSector1NetDev = lteHelper->InstallEnbDevice (enbSector1Container);

  //SECTOR 2 eNB configuration
  orientationDegrees = sector0AngleRad * 180.0 / M_PI + 120;
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (orientationDegrees));
  std::cout << "Sector 2 antenna orientation wrt x axis: " << orientationDegrees << " degrees" << std::endl;
  
  lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (centralFrequencyBand1Dl));
  lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (centralFrequencyBand1Ul));
  enbSector2NetDev = lteHelper->InstallEnbDevice (enbSector2Container);

  //SECTOR 3 eNB configuration
  orientationDegrees = sector0AngleRad * 180.0 / M_PI - 120;
  lteHelper->SetEnbAntennaModelAttribute ("Orientation", DoubleValue (orientationDegrees));
  std::cout << "Sector 3 antenna orientation wrt x axis: " << orientationDegrees << " degrees" << std::endl;
  
  lteHelper->SetEnbDeviceAttribute ("DlEarfcn", UintegerValue (centralFrequencyBand2Dl));
  lteHelper->SetEnbDeviceAttribute ("UlEarfcn", UintegerValue (centralFrequencyBand2Ul));
  enbSector3NetDev = lteHelper->InstallEnbDevice (enbSector3Container);

    /*
  ueSector1NetDev = lteHelper->InstallUeDevice (ueSector1Container);
  NetDeviceContainer ueNetDevs (ueSector1NetDev);
  ueSector2NetDev = lteHelper->InstallUeDevice (ueSector2Container);
  ueNetDevs.Add (ueSector2NetDev);
  ueSector3NetDev = lteHelper->InstallUeDevice (ueSector3Container);
  ueNetDevs.Add (ueSector3NetDev);*/
    
  int64_t randomStream = 1;
  randomStream += lteHelper->AssignStreams (enbSector1NetDev, randomStream);
  randomStream += lteHelper->AssignStreams (enbSector2NetDev, randomStream);
  randomStream += lteHelper->AssignStreams (enbSector3NetDev, randomStream);
  
  //randomStream += lteHelper->AssignStreams (ueSector1NetDev, randomStream);
  //randomStream += lteHelper->AssignStreams (ueSector2NetDev, randomStream);
  //randomStream += lteHelper->AssignStreams (ueSector3NetDev, randomStream);

/*
  ueNetDevsContainer = lteHelper->InstallUeDevice (ueNodesContainer);  
  int64_t randomStream = 1;
  randomStream += lteHelper->AssignStreams (ueNetDevsContainer, randomStream);    
  // This part is for asserts or sanity checks   
  for (auto nd = ueNetDevsContainer.Begin (); nd != ueNetDevsContainer.End (); ++nd)
    {
      auto ueNetDevice = DynamicCast<LteUeNetDevice> (*nd);
      NS_ASSERT (ueNetDevice->GetCcMap ().size () == 1);
      auto uePhy = ueNetDevice->GetPhy ();
    }*/
}

} // namespace ns3
