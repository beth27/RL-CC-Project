/* -*-  Mode: C++; c-file-style: "gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2018 Piotr Gawlowicz
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Piotr Gawlowicz <gawlowicz.p@gmail.com>
 * Based on script: ./examples/tcp/tcp-variants-comparison.cc
 *
 * Topology:
 *
 *   Right Leafs (Clients)                      Left Leafs (Sinks)
 *           |            \                    /        |
 *           |             \    bottleneck    /         |
 *           |              R0--------------R1          |
 *           |             /                  \         |
 *           |   access   /                    \ access |
 *           N -----------                      --------N
 */

#include <iostream>
#include <fstream>
#include <string>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/applications-module.h"
#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/enum.h"
#include "ns3/event-id.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"
#include "ns3/netanim-module.h"
#include "ns3/opengym-module.h"
#include "tcp-rl.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("TcpVariantsComparison");

static std::vector<uint32_t> rxPkts;

/*
static std::map<uint32_t, bool> firstCwnd;
static std::map<uint32_t, bool> firstSshThr;
static std::map<uint32_t, bool> firstRtt;
static std::map<uint32_t, bool> firstRto;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> cWndStream;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> ssThreshStream;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> rttStream;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> rtoStream;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> nextTxStream;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> nextRxStream;
static std::map<uint32_t, Ptr<OutputStreamWrapper>> inFlightStream;
static std::map<uint32_t, uint32_t> cWndValue;
static std::map<uint32_t, uint32_t> ssThreshValue;

static uint32_t
GetNodeIdFromContext (std::string context)
{
  std::size_t const n1 = context.find_first_of ("/", 1);
  std::size_t const n2 = context.find_first_of ("/", n1 + 1);
  return std::stoul (context.substr (n1 + 1, n2 - n1 - 1));
}

static void
CwndTracer (std::string context,  uint32_t oldval, uint32_t newval)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  if (firstCwnd[nodeId])
    {
      *cWndStream[nodeId]->GetStream () << "0.0 " << oldval << std::endl;
      firstCwnd[nodeId] = false;
    }
  *cWndStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << newval << std::endl;
  cWndValue[nodeId] = newval;

  if (!firstSshThr[nodeId])
    {
      *ssThreshStream[nodeId]->GetStream ()
          << Simulator::Now ().GetSeconds () << " " << ssThreshValue[nodeId] << std::endl;
    }
}

static void
SsThreshTracer (std::string context, uint32_t oldval, uint32_t newval)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  if (firstSshThr[nodeId])
    {
      *ssThreshStream[nodeId]->GetStream () << "0.0 " << oldval << std::endl;
      firstSshThr[nodeId] = false;
    }
  *ssThreshStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << newval << std::endl;
  ssThreshValue[nodeId] = newval;

  if (!firstCwnd[nodeId])
    {
      *cWndStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << cWndValue[nodeId] << std::endl;
    }
}

static void
RttTracer (std::string context, Time oldval, Time newval)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  if (firstRtt[nodeId])
    {
      *rttStream[nodeId]->GetStream () << "0.0 " << oldval.GetSeconds () << std::endl;
      firstRtt[nodeId] = false;
    }
  *rttStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << newval.GetSeconds () << std::endl;
}

static void
RtoTracer (std::string context, Time oldval, Time newval)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  if (firstRto[nodeId])
    {
      *rtoStream[nodeId]->GetStream () << "0.0 " << oldval.GetSeconds () << std::endl;
      firstRto[nodeId] = false;
    }
  *rtoStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << newval.GetSeconds () << std::endl;
}

static void
NextTxTracer (std::string context, [[maybe_unused]] SequenceNumber32 old, SequenceNumber32 nextTx)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  *nextTxStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << nextTx << std::endl;
}

static void
InFlightTracer (std::string context, [[maybe_unused]] uint32_t old, uint32_t inFlight)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  *inFlightStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << inFlight << std::endl;
}

static void
NextRxTracer (std::string context, [[maybe_unused]] SequenceNumber32 old, SequenceNumber32 nextRx)
{
  uint32_t nodeId = GetNodeIdFromContext (context);

  *nextRxStream[nodeId]->GetStream () << Simulator::Now ().GetSeconds () << " " << nextRx << std::endl;
}

static void
TraceCwnd (std::string cwnd_tr_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  cWndStream[nodeId] = ascii.CreateFileStream (cwnd_tr_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) + "/$ns3::TcpL4Protocol/SocketList/0/CongestionWindow",
                   MakeCallback (&CwndTracer));
}

static void
TraceSsThresh (std::string ssthresh_tr_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  ssThreshStream[nodeId] = ascii.CreateFileStream (ssthresh_tr_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) + "/$ns3::TcpL4Protocol/SocketList/0/SlowStartThreshold",
                   MakeCallback (&SsThreshTracer));
}

static void
TraceRtt (std::string rtt_tr_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  rttStream[nodeId] = ascii.CreateFileStream (rtt_tr_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) + "/$ns3::TcpL4Protocol/SocketList/0/RTT",
                   MakeCallback (&RttTracer));
}

static void
TraceRto (std::string rto_tr_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  rtoStream[nodeId] = ascii.CreateFileStream (rto_tr_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) + "/$ns3::TcpL4Protocol/SocketList/0/RTO",
                   MakeCallback (&RtoTracer));
}

static void
TraceNextTx (std::string &next_tx_seq_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  nextTxStream[nodeId] = ascii.CreateFileStream (next_tx_seq_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) + "/$ns3::TcpL4Protocol/SocketList/0/NextTxSequence",
                   MakeCallback (&NextTxTracer));
}

static void
TraceInFlight (std::string &in_flight_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  inFlightStream[nodeId] = ascii.CreateFileStream (in_flight_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) + "/$ns3::TcpL4Protocol/SocketList/0/BytesInFlight",
                   MakeCallback (&InFlightTracer));
}

static void
TraceNextRx (std::string &next_rx_seq_file_name, uint32_t nodeId)
{
  AsciiTraceHelper ascii;
  nextRxStream[nodeId] = ascii.CreateFileStream (next_rx_seq_file_name.c_str ());
  Config::Connect ("/NodeList/" + std::to_string (nodeId) +
                       "/$ns3::TcpL4Protocol/SocketList/1/RxBuffer/NextRxSequence",
                   MakeCallback (&NextRxTracer));
}
*/

static void
CountRxPkts(uint32_t sinkId, Ptr<const Packet> packet, const Address & srcAddr)
{
  rxPkts[sinkId]++;
}

static void
PrintRxCount()
{
  uint32_t size = rxPkts.size();
  NS_LOG_UNCOND("RxPkts:");
  for (uint32_t i=0; i<size; i++){
    NS_LOG_UNCOND("---SinkId: "<< i << " RxPkts: " << rxPkts.at(i));
  }
}

int main (int argc, char *argv[])
{

  //LogComponentEnable ("Config", LOG_LEVEL_ALL);

  uint32_t openGymPort = 5555;
  double tcpEnvTimeStep = 0.1;

  uint32_t nLeaf = 1;
  std::string transport_prot = "TcpRl";
  double error_p = 0.0;
  std::string bottleneck_bandwidth = "2Mbps";
  std::string bottleneck_delay = "0.01ms";
  std::string access_bandwidth = "10Mbps";
  std::string access_delay = "20ms";
  std::string prefix_file_name = "TcpVariantsComparison";
  uint64_t data_mbytes = 0;
  uint32_t mtu_bytes = 400;
  double duration = 60.0;
  uint32_t run = 0;
  bool sack = true;
  std::string queue_disc_type = "ns3::PfifoFastQueueDisc";
  std::string recovery = "ns3::TcpClassicRecovery";
  bool tracing = false;
  bool flow_monitor = false;
  bool pcap = true;    

  CommandLine cmd;
  // required parameters for OpenGym interface
  cmd.AddValue ("openGymPort", "Port number for OpenGym env. Default: 5555", openGymPort);
  cmd.AddValue ("simSeed", "Seed for random generator. Default: 1", run);
  cmd.AddValue ("envTimeStep", "Time step interval for time-based TCP env [s]. Default: 0.1s", tcpEnvTimeStep);
  // other parameters
  cmd.AddValue ("nLeaf",     "Number of left and right side leaf nodes", nLeaf);
  cmd.AddValue ("transport_prot", "Transport protocol to use: TcpNewReno, "
                "TcpHybla, TcpHighSpeed, TcpHtcp, TcpVegas, TcpScalable, TcpVeno, "
                "TcpBic, TcpYeah, TcpIllinois, TcpWestwood, TcpWestwoodPlus, TcpLedbat, "
		            "TcpLp, TcpRl, TcpRlTimeBased", transport_prot);
  cmd.AddValue ("error_p", "Packet error rate", error_p);
  cmd.AddValue ("bottleneck_bandwidth", "Bottleneck bandwidth", bottleneck_bandwidth);
  cmd.AddValue ("bottleneck_delay", "Bottleneck delay", bottleneck_delay);
  cmd.AddValue ("access_bandwidth", "Access link bandwidth", access_bandwidth);
  cmd.AddValue ("access_delay", "Access link delay", access_delay);
  cmd.AddValue ("prefix_name", "Prefix of output trace file", prefix_file_name);
  cmd.AddValue ("data", "Number of Megabytes of data to transmit", data_mbytes);
  cmd.AddValue ("mtu", "Size of IP packets to send in bytes", mtu_bytes);
  cmd.AddValue ("duration", "Time to allow flows to run in seconds", duration);
  cmd.AddValue ("run", "Run index (for setting repeatable seeds)", run);
  cmd.AddValue ("queue_disc_type", "Queue disc type for gateway (e.g. ns3::CoDelQueueDisc)", queue_disc_type);
  cmd.AddValue ("sack", "Enable or disable SACK option", sack);
  cmd.AddValue ("recovery", "Recovery algorithm type to use (e.g., ns3::TcpPrrRecovery", recovery);
  cmd.AddValue ("tracing", "Flag to enable/disable tracing", tracing);
  cmd.AddValue ("flow_monitor", "Enable flow monitor", flow_monitor);
  cmd.AddValue ("pcap_tracing", "Enable or disable PCAP tracing", pcap);

  cmd.Parse (argc, argv);

  transport_prot = std::string ("ns3::") + transport_prot;

  SeedManager::SetSeed (1);
  SeedManager::SetRun (run);

  NS_LOG_UNCOND("Ns3Env parameters:");
  if (transport_prot.compare ("ns3::TcpRl") == 0 or transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    NS_LOG_UNCOND("--openGymPort: " << openGymPort);
  } else {
    NS_LOG_UNCOND("--openGymPort: No OpenGym");
  }

  NS_LOG_UNCOND("--seed: " << run);
  NS_LOG_UNCOND("--Tcp version: " << transport_prot);
  NS_LOG_UNCOND("--duration: " << duration);

  // OpenGym Env --- has to be created before any other thing
  Ptr<OpenGymInterface> openGymInterface;
  if (transport_prot.compare ("ns3::TcpRl") == 0)
  {
    openGymInterface = OpenGymInterface::Get(openGymPort);
    Config::SetDefault ("ns3::TcpRl::Reward", DoubleValue (1.0)); // Reward when increasing congestion window
    Config::SetDefault ("ns3::TcpRl::Penalty", DoubleValue (-2.0)); // Penalty when loss occurs
  }

  if (transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    openGymInterface = OpenGymInterface::Get(openGymPort);
    Config::SetDefault ("ns3::TcpRlTimeBased::StepTime", TimeValue (Seconds(tcpEnvTimeStep))); // Time step of TCP env
    Config::SetDefault ("ns3::TcpRl::Reward", DoubleValue (2.0)); // Reward when increasing congestion window
    Config::SetDefault ("ns3::TcpRl::Penalty", DoubleValue (-30.0)); // Penalty when loss occurs
  }

  // Calculate the ADU size
  Header* temp_header = new Ipv4Header ();
  uint32_t ip_header = temp_header->GetSerializedSize ();
  NS_LOG_LOGIC ("IP Header size is: " << ip_header);
  delete temp_header;
  temp_header = new TcpHeader ();
  uint32_t tcp_header = temp_header->GetSerializedSize ();
  NS_LOG_LOGIC ("TCP Header size is: " << tcp_header);
  delete temp_header;
  uint32_t tcp_adu_size = mtu_bytes - 20 - (ip_header + tcp_header);
  NS_LOG_LOGIC ("TCP ADU size is: " << tcp_adu_size);

  // Set the simulation start and stop time
  double start_time = 0.1;
  double stop_time = start_time + duration;

  // 4 MB of TCP buffer
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 21));
  Config::SetDefault ("ns3::TcpSocketBase::Sack", BooleanValue (sack));
  Config::SetDefault ("ns3::TcpSocket::DelAckCount", UintegerValue (2));


  Config::SetDefault ("ns3::TcpL4Protocol::RecoveryType",
                      TypeIdValue (TypeId::LookupByName (recovery)));
  // Select TCP variant
  if (transport_prot.compare ("ns3::TcpWestwoodPlus") == 0)
    {
      // TcpWestwoodPlus is not an actual TypeId name; we need TcpWestwood here
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TcpWestwood::GetTypeId ()));
      // the default protocol type in ns3::TcpWestwood is WESTWOOD
      Config::SetDefault ("ns3::TcpWestwood::ProtocolType", EnumValue (TcpWestwood::WESTWOODPLUS));
    }
  else
    {
      TypeId tcpTid;
      NS_ABORT_MSG_UNLESS (TypeId::LookupByNameFailSafe (transport_prot, &tcpTid), "TypeId " << transport_prot << " not found");
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType", TypeIdValue (TypeId::LookupByName (transport_prot)));
    }

  // Configure the error model
  // Here we use RateErrorModel with packet error rate
  Ptr<UniformRandomVariable> uv = CreateObject<UniformRandomVariable> ();
  uv->SetStream (50);
  RateErrorModel error_model;
  error_model.SetRandomVariable (uv);
  error_model.SetUnit (RateErrorModel::ERROR_UNIT_PACKET);
  error_model.SetRate (error_p);

  // Create the point-to-point link helpers
  PointToPointHelper bottleNeckLink;
  bottleNeckLink.SetDeviceAttribute  ("DataRate", StringValue (bottleneck_bandwidth));
  bottleNeckLink.SetChannelAttribute ("Delay", StringValue (bottleneck_delay));
  //bottleNeckLink.SetDeviceAttribute  ("ReceiveErrorModel", PointerValue (&error_model));

  PointToPointHelper pointToPointLeaf;
  pointToPointLeaf.SetDeviceAttribute  ("DataRate", StringValue (access_bandwidth));
  pointToPointLeaf.SetChannelAttribute ("Delay", StringValue (access_delay));

  PointToPointDumbbellHelper d (nLeaf, pointToPointLeaf,
                                nLeaf, pointToPointLeaf,
                                bottleNeckLink);

  // Install IP stack
  InternetStackHelper stack;
  stack.InstallAll ();

  // Traffic Control
  TrafficControlHelper tchPfifo;
  tchPfifo.SetRootQueueDisc ("ns3::PfifoFastQueueDisc");

  TrafficControlHelper tchCoDel;
  tchCoDel.SetRootQueueDisc ("ns3::CoDelQueueDisc");

  DataRate access_b (access_bandwidth);
  DataRate bottle_b (bottleneck_bandwidth);
  Time access_d (access_delay);
  Time bottle_d (bottleneck_delay);

  uint32_t size = static_cast<uint32_t>((std::min (access_b, bottle_b).GetBitRate () / 8) *
    ((access_d + bottle_d + access_d) * 2).GetSeconds ());

  Config::SetDefault ("ns3::PfifoFastQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::PACKETS, size / mtu_bytes)));
  Config::SetDefault ("ns3::CoDelQueueDisc::MaxSize",
                      QueueSizeValue (QueueSize (QueueSizeUnit::BYTES, size)));

  NS_LOG_UNCOND("Router buffer size: "<<size<<" bytes");

  if (queue_disc_type.compare ("ns3::PfifoFastQueueDisc") == 0)
  {
    tchPfifo.Install (d.GetLeft()->GetDevice(1));
    tchPfifo.Install (d.GetRight()->GetDevice(1));
  }
  else if (queue_disc_type.compare ("ns3::CoDelQueueDisc") == 0)
  {
    tchCoDel.Install (d.GetLeft()->GetDevice(1));
    tchCoDel.Install (d.GetRight()->GetDevice(1));
  }
  else
  {
    NS_FATAL_ERROR ("Queue not recognized. Allowed values are ns3::CoDelQueueDisc or ns3::PfifoFastQueueDisc");
  }

  // Assign IP Addresses
  d.AssignIpv4Addresses (Ipv4AddressHelper (Ipv4Address("10.1.1.0"), Ipv4Mask("255.255.255.0"),Ipv4Address("0.0.0.1")),
                         Ipv4AddressHelper (Ipv4Address("10.2.1.0"), Ipv4Mask("255.255.255.0"),Ipv4Address("0.0.0.1")),
                         Ipv4AddressHelper (Ipv4Address("10.3.1.0"), Ipv4Mask("255.255.255.0"),Ipv4Address("0.0.0.1")));


  NS_LOG_INFO ("Initialize Global Routing.");
  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  // Install apps in left and right nodes
  uint16_t port = 50000;
  Address sinkLocalAddress (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sinkHelper ("ns3::TcpSocketFactory", sinkLocalAddress);
  ApplicationContainer sinkApps;
  for (uint32_t i = 0; i < d.RightCount (); ++i)
  {
    sinkHelper.SetAttribute ("Protocol", TypeIdValue (TcpSocketFactory::GetTypeId ()));
    sinkApps.Add (sinkHelper.Install (d.GetRight (i)));
  }
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop  (Seconds (stop_time));

  for (uint32_t i = 0; i < d.LeftCount (); ++i)
  {
    // Create an on/off app sending packets to the left side
    AddressValue remoteAddress (InetSocketAddress (d.GetRightIpv4Address (i), port));
    //NS_LOG_UNCOND("ipv4: "<<d.GetRightIpv4Address (i));
    Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (tcp_adu_size));
    BulkSendHelper ftp ("ns3::TcpSocketFactory", Address ());
    ftp.SetAttribute ("Remote", remoteAddress);
    ftp.SetAttribute ("SendSize", UintegerValue (tcp_adu_size));
    ftp.SetAttribute ("MaxBytes", UintegerValue (data_mbytes * 1000000));

    NS_LOG_UNCOND("---ftp sendsize: " << tcp_adu_size);
    NS_LOG_UNCOND("---ftp maxbytes: " << data_mbytes * 1000000);

    ApplicationContainer clientApp = ftp.Install (d.GetLeft (i));
    clientApp.Start (Seconds (start_time * i)); // Start after sink
    clientApp.Stop (Seconds (stop_time - 3)); // Stop before the sink
  }



  // Node path not recognized whiile trying to connect. Point to point dumb bell helper is used
  /*if (tracing)
    {
      std::ofstream ascii;
      Ptr<OutputStreamWrapper> ascii_wrap;
      ascii.open ((prefix_file_name + "-ascii").c_str ());
      ascii_wrap = new OutputStreamWrapper ((prefix_file_name + "-ascii").c_str (),
                                            std::ios::out);
      stack.EnableAsciiIpv4All (ascii_wrap);

      for (uint16_t index = 0; index < nLeaf; index++)
        {
          std::string flowString ("");
          if (nLeaf>1)
            {
              flowString = "-flow" + std::to_string (index);
            }

          firstCwnd[index+1] = true;
          firstSshThr[index+1] = true;
          firstRtt[index+1] = true;
          firstRto[index+1] = true;

          Simulator::Schedule (Seconds (start_time * index + 0.00001), &TraceCwnd,
                               prefix_file_name + flowString + "-cwnd.data", index+1);
          Simulator::Schedule (Seconds (start_time * index + 0.00001), &TraceSsThresh,
                               prefix_file_name + flowString + "-ssth.data", index+1);
          Simulator::Schedule (Seconds (start_time * index + 0.00001), &TraceRtt,
                               prefix_file_name + flowString + "-rtt.data", index+1);
          Simulator::Schedule (Seconds (start_time * index + 0.00001), &TraceRto,
                               prefix_file_name + flowString + "-rto.data", index+1);
          Simulator::Schedule (Seconds (start_time * index + 0.00001), &TraceNextTx,
                               prefix_file_name + flowString + "-next-tx.data", index+1);
          Simulator::Schedule (Seconds (start_time * index + 0.00001), &TraceInFlight,
                               prefix_file_name + flowString + "-inflight.data", index+1);
          Simulator::Schedule (Seconds (start_time * index + 0.1), &TraceNextRx,
                               prefix_file_name + flowString + "-next-rx.data", nLeaf+index+1);
        }
    }*/

  if (pcap)
    {
      bottleNeckLink.EnablePcapAll (prefix_file_name, true);
      pointToPointLeaf.EnablePcapAll (prefix_file_name, true);
    }

  // Flow monitor
  FlowMonitorHelper flowHelper;
  if (true)
    {
      flowHelper.InstallAll ();
    }

  // Count RX packets
  for (uint32_t i = 0; i < d.RightCount (); ++i)
  {
    rxPkts.push_back(0);
    Ptr<PacketSink> pktSink = DynamicCast<PacketSink>(sinkApps.Get(i));
    pktSink->TraceConnectWithoutContext ("Rx", MakeBoundCallback (&CountRxPkts, i));
  }

  Simulator::Stop (Seconds (stop_time));

  //AnimationInterface anim("anim1.xml");
  //anim.SetConstantPosition(d.GetRight (0),1.0,2.0);
  //anim.SetConstantPosition(d.GetLeft (0),2.0,3.0);

  Simulator::Run ();

  if (flow_monitor)
    {
      flowHelper.SerializeToXmlFile (prefix_file_name + ".flowmonitor", true, true);
    }

  if (transport_prot.compare ("ns3::TcpRl") == 0 or transport_prot.compare ("ns3::TcpRlTimeBased") == 0)
  {
    openGymInterface->NotifySimulationEnd();
  }

  PrintRxCount();
  Simulator::Destroy ();
  return 0;
}
