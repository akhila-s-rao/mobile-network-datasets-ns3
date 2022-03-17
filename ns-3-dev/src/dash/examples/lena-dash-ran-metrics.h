using namespace ns3;

AsciiTraceHelper traceHelper;
Ptr<OutputStreamWrapper> mobStream = traceHelper.CreateFileStream ("mobility_trace.txt");
Ptr<OutputStreamWrapper> parmStream = traceHelper.CreateFileStream ("parameter_settings.txt");
//Ptr<OutputStreamWrapper> rsrpRsrqStream = traceHelper.CreateFileStream ("rsrp_rsrq.txt");
//Ptr<OutputStreamWrapper> rlcBufferSizeStream = traceHelper.CreateFileStream ("rlc_buffer.txt");
//Ptr<OutputStreamWrapper> dashClientStream = traceHelper.CreateFileStream ("dash_client.txt");
/*
uint16_t
GetNodeIdFromContext(std::string context){
  std::string path = context.substr (10, context.length());
  std::string nodeIdStr = path.substr (0, path.find ("/"));
  uint16_t nodeId = stoi(nodeIdStr);
  return (nodeId);
}

uint16_t
GetNodeIdFromLteCellId (uint16_t cellId)
{
  uint16_t nodeId = 0;
  nodeId = cellId; //+1 is for remote host
  return (nodeId);
}
*/

/*
void // Trace at UE
Lte_ReportUeMeasurements (Ptr<OutputStreamWrapper> stream,
              std::string context, 
              uint16_t rnti,
              uint16_t cellId, 
              double avg_rsrp, 
              double avg_rsrq, 
              bool servingCell)
{
  *stream->GetStream()  << Simulator::Now ().GetMicroSeconds ()
        //<< " " << GetNodeIdFromContext(context) //ue_node_id 
        //<< " " << GetNodeIdFromLteCellId(cellId) //nbr_enb_node_id
        << "\t" << context 
        << "\t" << avg_rsrp //avg_rsrp_dBm 
        << "\t" << avg_rsrq // avg_rsrq_dBm 
        << "\t" << servingCell // isServingCell 
        << std::endl;
}
*/
void
Lte_RlcBufferSize (Ptr<OutputStreamWrapper> stream,
          std::string context,
          uint16_t rnti,
          uint8_t lcid,
          uint32_t txQueueSize,
          uint32_t retxQueueSize)
{
  //std::cout << context << endl;
 *stream->GetStream() << Simulator::Now ().GetMicroSeconds () 
    << "\t" << context 
    << "\t" << rnti 
    << "\t" << (uint16_t)lcid 
    << "\t" << txQueueSize 
    << "\t" << retxQueueSize << std::endl;
}
